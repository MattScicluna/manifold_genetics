version 1.0

## manifold-genetics as a WDL workflow (Cromwell, e.g. Verily Workbench).
##
## The same pipeline as main.nf: `manifold-genetics pipeline` in the project's
## container. Defaults are the published biobank settings (fit-set embedding,
## knn 500, t 50, 10,000 random landmarks); run_admixture adds admixture, its
## plots and metrics (on CPU: the image's torch is the CPU build). extra_args is appended last, so it can override any
## option (e.g. "--embedding umap"). Example inputs: wdl/inputs.*.json.

workflow manifold_genetics {
    input {
        # PLINK files (WDL takes files, not prefixes). project_* default to fit_*.
        File fit_bed
        File fit_bim
        File fit_fam
        File? project_bed
        File? project_bim
        File? project_fam

        # Labels for every sample, and the colormap.
        File labels
        File colormap
        File? geographic

        Boolean run_admixture = false
        Int n_pcs = 20
        Int n_components = 2
        String embedding_input = "fit"
        Int knn = 500
        Int t = 50
        Int n_landmark = 10000      # 0 = no landmarks (dense n x n memory)
        Int k_min = 2
        Int k_max = 10
        String extra_args = ""

        String docker
        Int cpu = 16
        Int memory_gb = 128
        Int disk_gb = 500
    }

    call pipeline {
        input:
            fit_bed = fit_bed,
            fit_bim = fit_bim,
            fit_fam = fit_fam,
            project_bed = select_first([project_bed, fit_bed]),
            project_bim = select_first([project_bim, fit_bim]),
            project_fam = select_first([project_fam, fit_fam]),
            labels = labels,
            colormap = colormap,
            geographic = geographic,
            run_admixture = run_admixture,
            n_pcs = n_pcs,
            n_components = n_components,
            embedding_input = embedding_input,
            knn = knn,
            t = t,
            n_landmark = n_landmark,
            k_min = k_min,
            k_max = k_max,
            extra_args = extra_args,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb
    }

    output {
        File results = pipeline.results
        Array[File] figures = pipeline.figures
    }
}

task pipeline {
    input {
        File fit_bed
        File fit_bim
        File fit_fam
        File project_bed
        File project_bim
        File project_fam
        File labels
        File colormap
        File? geographic
        Boolean run_admixture
        Int n_pcs
        Int n_components
        String embedding_input
        Int knn
        Int t
        Int n_landmark
        Int k_min
        Int k_max
        String extra_args
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail

        # Admixture's torch shares memory through a Unix socket under TMPDIR,
        # and a socket path is limited to ~104 bytes. Cromwell's per-task TMPDIR
        # can be longer, which fails as "no response from torch_shm_manager".
        if [ "${#TMPDIR}" -gt 60 ]; then
            TMPDIR="$(mktemp -d /tmp/mg.XXXXXX)"
            export TMPDIR
        fi

        # The pipeline takes PLINK prefixes: link each triple under one prefix,
        # since Cromwell may localise the three files to different directories.
        mkdir -p fit project
        ln -s "~{fit_bed}" fit/data.bed
        ln -s "~{fit_bim}" fit/data.bim
        ln -s "~{fit_fam}" fit/data.fam
        ln -s "~{project_bed}" project/data.bed
        ln -s "~{project_bim}" project/data.bim
        ln -s "~{project_fam}" project/data.fam

        manifold-genetics pipeline \
            --fit-plink fit/data \
            --project-plink project/data \
            --labels "~{labels}" \
            --colormap "~{colormap}" \
            ~{"--geographic " + geographic} \
            --output results \
            --threads ~{cpu} \
            --n-pcs ~{n_pcs} \
            --embedding phate \
            --embedding-input ~{embedding_input} \
            --n-components ~{n_components} \
            --knn ~{knn} \
            --t ~{t} \
            ~{if n_landmark > 0 then "--n-landmark " + n_landmark + " --random-landmarking" else ""} \
            ~{if run_admixture then "--k-min " + k_min + " --k-max " + k_max else "--skip-admixture"} \
            ~{extra_args}

        tar -czf results.tar.gz results
    >>>

    output {
        File results = "results.tar.gz"
        Array[File] figures = glob("results/figures/*/*.png")
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
    }
}
