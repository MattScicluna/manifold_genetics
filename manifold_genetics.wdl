version 1.0

## manifold-genetics as a WDL workflow (Cromwell, e.g. Verily Workbench).
##
## The same pipeline as main.nf: `manifold-genetics pipeline` in the project's
## container. WDL has no profiles, so run_admixture chooses between the two
## modes: true runs everything; false passes --skip-admixture (no admixture
## step, plots or metrics). Example inputs: wdl/inputs.full.json and
## wdl/inputs.no_admixture.json.

workflow manifold_genetics {
    input {
        # PLINK files (WDL takes files, not prefixes). project_* default to fit_*.
        File fit_bed
        File fit_bim
        File fit_fam
        File? project_bed
        File? project_bim
        File? project_fam

        # One labels CSV and colormap for both sets, or fit_/project_ versions.
        File? labels
        File? colormap
        File? fit_labels
        File? project_labels
        File? fit_colormap
        File? project_colormap
        File? geographic

        Boolean run_admixture = true

        Int n_pcs = 20
        String pca_backend = "python"
        String embedding = "phate"
        String embedding_input = "both"
        Int n_components = 2
        Int? knn
        String? t
        Int? n_landmark
        Boolean random_landmarking = false
        Int k_min = 2
        Int k_max = 10
        String extra_args = ""

        String docker
        Int cpu = 16
        Int memory_gb = 128
        Int disk_gb = 500
        Int gpus = 0
        String gpu_type = "nvidia-tesla-t4"
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
            fit_labels = fit_labels,
            project_labels = project_labels,
            fit_colormap = fit_colormap,
            project_colormap = project_colormap,
            geographic = geographic,
            run_admixture = run_admixture,
            n_pcs = n_pcs,
            pca_backend = pca_backend,
            embedding = embedding,
            embedding_input = embedding_input,
            n_components = n_components,
            knn = knn,
            t = t,
            n_landmark = n_landmark,
            random_landmarking = random_landmarking,
            k_min = k_min,
            k_max = k_max,
            extra_args = extra_args,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb,
            gpus = gpus,
            gpu_type = gpu_type
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
        File? labels
        File? colormap
        File? fit_labels
        File? project_labels
        File? fit_colormap
        File? project_colormap
        File? geographic
        Boolean run_admixture
        Int n_pcs
        String pca_backend
        String embedding
        String embedding_input
        Int n_components
        Int? knn
        String? t
        Int? n_landmark
        Boolean random_landmarking
        Int k_min
        Int k_max
        String extra_args
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
        Int gpus
        String gpu_type
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
            ~{"--labels " + labels} \
            ~{"--colormap " + colormap} \
            ~{"--fit-labels " + fit_labels} \
            ~{"--project-labels " + project_labels} \
            ~{"--fit-colormap " + fit_colormap} \
            ~{"--project-colormap " + project_colormap} \
            ~{"--geographic " + geographic} \
            --output results \
            --threads ~{cpu} \
            --n-pcs ~{n_pcs} \
            --pca-backend ~{pca_backend} \
            --embedding ~{embedding} \
            --embedding-input ~{embedding_input} \
            --n-components ~{n_components} \
            ~{"--knn " + knn} \
            ~{"--t " + t} \
            ~{"--n-landmark " + n_landmark} \
            ~{if random_landmarking then "--random-landmarking" else ""} \
            ~{if run_admixture then "--k-min " + k_min + " --k-max " + k_max else "--skip-admixture"} \
            ~{if run_admixture && gpus > 0 then "--num-gpus " + gpus else ""} \
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
        gpuCount: gpus
        gpuType: gpu_type
        preemptible: 0
    }
}
