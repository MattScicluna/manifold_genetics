version 1.0

## Admixture for experiments analyse.wdl has run (optional; run after it).
## Each experiment's fit set is rebuilt from its published data/fit_samples.txt,
## Neural Admixture is trained on it for K in [k_min, k_max], and the Q matrices
## go to <output_dir>/<name>/outputs/admixture/ (fit.<K>.csv; project.<K>.csv too
## with infer_project, which needs ~1 byte per genotype of the whole cohort in
## memory, e.g. 75 GB for UK Biobank).
## Figures: the bar chart, and every embedding already published under
## outputs/embeddings/ coloured by admixture (analyse's 2-D, analyse_3d's 3-D);
## nothing is re-embedded.
##
## Runs on a GPU, in the -gpu image (Neural Admixture on CPU takes days at
## biobank size); the workspace needs GPU quota for gpu_type.

workflow admixture {
    input {
        # The prepared cohort analyse.wdl ran on, e.g. gs://<bucket>/aou_v9/prepared
        String cohort_dir
        # analyse.wdl's output_dir, e.g. gs://<bucket>/aou_v9
        String output_dir
        # analyse.wdl's experiment names.
        Array[String] experiments = ["balanced", "geosketch_90k"]
        Int k_min = 2
        Int k_max = 10
        # Also infer the project set (the whole cohort); see memory above.
        Boolean infer_project = false

        String docker = "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/mattscicluna/manifold-genetics:verily-workflow-gpu"
        Int gpus = 1
        String gpu_type = "nvidia-tesla-t4"
        Int cpu = 8
        Int memory_gb = 64
        Int disk_gb = 200
    }

    scatter (name in experiments) {
        call admixture_task {
            input:
                name = name,
                config = cohort_dir + "/config.yaml",
                colormap_fit = cohort_dir + "/colormap_fit.json",
                colormap_project = cohort_dir + "/colormap_project.json",
                fit_bed = cohort_dir + "/data/fit_subset.bed",
                fit_bim = cohort_dir + "/data/fit_subset.bim",
                fit_fam = cohort_dir + "/data/fit_subset.fam",
                fit_labels = cohort_dir + "/data/fit_labels.csv",
                project_bed = cohort_dir + "/data/project_subset.bed",
                project_bim = cohort_dir + "/data/project_subset.bim",
                project_fam = cohort_dir + "/data/project_subset.fam",
                project_labels = cohort_dir + "/data/project_labels.csv",
                fit_samples = output_dir + "/" + name + "/data/fit_samples.txt",
                k_min = k_min,
                k_max = k_max,
                infer_project = infer_project,
                output_dir = output_dir,
                docker = docker,
                gpus = gpus,
                gpu_type = gpu_type,
                cpu = cpu,
                memory_gb = memory_gb,
                disk_gb = disk_gb
        }
    }

    output {
        Array[File] q_files = flatten(admixture_task.q_files)
        Array[File] figures = flatten(admixture_task.figures)
    }
}

task admixture_task {
    input {
        String name
        File config
        File colormap_fit
        File colormap_project
        File fit_bed
        File fit_bim
        File fit_fam
        File fit_labels
        File project_bed
        File project_bim
        File project_fam
        File project_labels
        File fit_samples
        Int k_min
        Int k_max
        Boolean infer_project
        String output_dir
        String docker
        Int gpus
        String gpu_type
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        exec > >(tee admixture.log) 2>&1

        # The cohort directory, rebuilt as in analyse.wdl.
        mkdir -p cohort/data
        cp "~{config}" cohort/config.yaml
        cp "~{colormap_fit}" cohort/colormap_fit.json
        cp "~{colormap_project}" cohort/colormap_project.json
        cp "~{fit_labels}" cohort/data/fit_labels.csv
        cp "~{project_labels}" cohort/data/project_labels.csv
        ln -s "~{fit_bed}" cohort/data/fit_subset.bed
        ln -s "~{fit_bim}" cohort/data/fit_subset.bim
        ln -s "~{fit_fam}" cohort/data/fit_subset.fam
        ln -s "~{project_bed}" cohort/data/project_subset.bed
        ln -s "~{project_bim}" cohort/data/project_subset.bim
        ln -s "~{project_fam}" cohort/data/project_subset.fam

        # The experiment's fit set, exactly as analyse chose it.
        manifold-genetics subsample cohort/config.yaml --out exp --fit-samples "~{fit_samples}"

        a=exp/outputs/admixture
        if [ "~{infer_project}" = "true" ]; then
            project=(--project-plink exp/data/project_subset --project-output "${a}/project")
            bars_q="${a}/project"; bars_labels=exp/data/project_labels.csv; bars_cmap=exp/colormap_project.json
        else
            # The CLI always infers a second set; without the project set, that is the fit set again, unpublished.
            project=(--project-plink exp/data/fit_subset --project-output scratch/fit_again)
            bars_q="${a}/fit"; bars_labels=exp/data/fit_labels.csv; bars_cmap=exp/colormap_fit.json
        fi
        manifold-genetics admixture --fit-plink exp/data/fit_subset "${project[@]}" \
            --neuraladmixture-output-dir "${a}/checkpoints" --fit-output "${a}/fit" \
            --k-min ~{k_min} --k-max ~{k_max} --threads ~{cpu} --num-gpus ~{gpus} \
            --neuraladmixture-batch-size 400

        # Figures: the bar chart (300 per group, as the pipeline draws it; of the
        # project set when inferred), then each published embedding coloured by
        # the fit set's proportions (the embeddings are of the fit set).
        f=exp/outputs/figures/admixture
        mkdir -p "${f}" exp/embeddings
        group=$(python3 -c "import json,sys; print(next(iter(json.load(open(sys.argv[1])))))" "${bars_cmap}")
        manifold-genetics plot-admixture --q-prefix "${bars_q}" --labels "${bars_labels}" \
            --group-column "${group}" --colormap "${bars_cmap}" --k-min ~{k_min} --k-max ~{k_max} \
            --subsample-per-group 300 --output "${f}/$(basename "${bars_q}")_bars.png" \
            --component-colors-output "${a}/component_colors.json"
        src="~{output_dir}/~{name}/outputs/embeddings"
        case "${src}" in
            gs://*) gcloud storage cp "${src}/*.csv" exp/embeddings/ || true ;;
            *) cp "${src}"/*.csv exp/embeddings/ 2>/dev/null || true ;;
        esac
        for emb in exp/embeddings/*.csv; do
            [ -e "${emb}" ] || { echo "No embeddings under ${src}; admixture-coloured embeddings skipped."; break; }
            manifold-genetics plot-admixture-embedding --embedding "${emb}" --q-prefix "${a}/fit" \
                --k-min ~{k_min} --k-max ~{k_max} --component-colormap "${a}/component_colors.json" \
                --output "${f}/$(basename "${emb}" .csv)_admixture.png" --no-hover-ids
        done

        mkdir -p publish/outputs
        cp -r "${a}" "${f%/admixture}" publish/outputs/
        rm -rf publish/outputs/admixture/checkpoints
        cp admixture.log publish/
        dest="~{output_dir}/~{name}"
        case "${dest}" in
            gs://*) gcloud storage rsync --recursive publish "${dest}" ;;
            *) mkdir -p "${dest}" && cp -R publish/. "${dest}/" ;;
        esac

    >>>

    output {
        # glob, as in analyse.wdl: Google Batch copies back only named outputs.
        Array[File] q_files = glob("exp/outputs/admixture/*.csv")
        Array[File] figures = glob("exp/outputs/figures/admixture/*")
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
        # Cromwell (Google Batch) reads gpuCount/gpuType; miniwdl reads gpu.
        gpuCount: gpus
        gpuType: gpu_type
        gpu: true
    }
}
