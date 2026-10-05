version 1.0

## Step 2 of 2: analyse a prepared cohort (any cohort directory; for All of Us,
## the <output_dir>/prepared/ that aou_prepare.wdl writes).
##
## Each experiment chooses a fit set with `manifold-genetics subsample` and runs
## two tasks: subsample + PCA (fit on the subset, project everyone), then the
## 2-D PHATE, UMAP and PCA figures of the fit set. Experiments run side by side;
## each publishes to <output_dir>/<name>/. The embeddings are their own task so
## changing their settings does not redo the PCA (with call caching).
## analyse_3d.wdl draws the 3-D versions from the PCA published here.
##
## The default experiments are the manuscript's All of Us fit sets.

struct Experiment {
    String name
    String subsample_args
}

workflow analyse {
    input {
        # The prepared cohort directory, e.g. gs://<bucket>/aou_v9/prepared
        String cohort_dir
        # Where results go, e.g. gs://<bucket>/aou_v9 (writes <output_dir>/<experiment>/).
        String output_dir

        Array[Experiment] experiments = [
            {
                "name": "balanced",
                "subsample_args": "--group 'race_ethnicity=White|European:10000' --group 'race_ethnicity=Black or African American:10000' --group 'race_ethnicity=Hispanic or Latino:10000' --group 'race_ethnicity=^Asian$:10000' --include-rest --seed 42"
            },
            {
                "name": "geosketch_90k",
                "subsample_args": "--geosketch 90000 --seed 42"
            }
        ]

        # UMAP (PHATE takes the subsample preset's settings from the config).
        Int umap_n_neighbors = 15
        Float umap_min_dist = 0.5

        String docker = "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/mattscicluna/manifold-genetics:verily-workflow"
        Int pca_cpu = 16
        Int pca_memory_gb = 128
        Int pca_disk_gb = 200
        Int embed_cpu = 8
        Int embed_memory_gb = 64
        Int embed_disk_gb = 50
    }

    scatter (experiment in experiments) {
        call subsample_pca {
            input:
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
                name = experiment.name,
                subsample_args = experiment.subsample_args,
                output_dir = output_dir,
                docker = docker,
                cpu = pca_cpu,
                memory_gb = pca_memory_gb,
                disk_gb = pca_disk_gb
        }

        call embed {
            input:
                name = experiment.name,
                config = subsample_pca.out_config,
                colormap_fit = subsample_pca.out_colormap_fit,
                colormap_project = subsample_pca.out_colormap_project,
                fit_labels = subsample_pca.out_fit_labels,
                project_labels = subsample_pca.out_project_labels,
                pca_csvs = subsample_pca.pca_csvs,
                umap_n_neighbors = umap_n_neighbors,
                umap_min_dist = umap_min_dist,
                output_dir = output_dir,
                docker = docker,
                cpu = embed_cpu,
                memory_gb = embed_memory_gb,
                disk_gb = embed_disk_gb
        }
    }

    output {
        Array[File] pca = flatten(subsample_pca.pca_csvs)
        Array[File] embeddings = flatten(embed.embeddings)
        Array[File] figures = flatten(embed.figures)
    }
}

task subsample_pca {
    input {
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
        String name
        String subsample_args
        String output_dir
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        exec > >(tee subsample_pca.log) 2>&1

        # The cohort directory, rebuilt where its config expects its files:
        # small files copied, PLINK sets linked.
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

        # A memory budget for the PCA, leaving headroom for the process.
        budget=$(( ~{memory_gb} * 2 / 5 ))

        # The experiment's subsample arguments, verbatim: they carry quoted
        # values ('race_ethnicity=Black or African American:10000', '^Asian$')
        # that must reach the shell's parser intact, so they are read through a
        # quoted heredoc and evaluated once, inside double quotes.
        read -r -d '' ARGS <<'SUBSAMPLE_ARGS' || true
        ~{subsample_args}
        SUBSAMPLE_ARGS
        extra=""
        case " ${ARGS} " in *" --geosketch "*) extra="--memory-gb ${budget}" ;; esac
        eval "manifold-genetics subsample cohort/config.yaml --out '~{name}' ${ARGS} ${extra}"
        manifold-genetics run "~{name}/config.yaml" --skip-embedding --skip-metrics --skip-admixture \
            --memory-gb "${budget}"

        mkdir -p publish/data publish/outputs
        cp "~{name}"/config.yaml "~{name}"/colormap_*.json publish/
        cp "~{name}"/data/*_labels.csv publish/data/
        for f in data/fit_samples.txt sketch_pca.csv; do
            if [ -f "~{name}/${f}" ]; then cp "~{name}/${f}" "publish/${f}"; fi
        done
        cp -r "~{name}/outputs/pca" "~{name}/outputs/figures" publish/outputs/
        cp subsample_pca.log publish/
        dest="~{output_dir}/~{name}"
        case "${dest}" in
            gs://*) gcloud storage rsync --recursive publish "${dest}" ;;
            *) mkdir -p "${dest}" && cp -R publish/. "${dest}/" ;;
        esac

    >>>

    output {
        File out_config = "~{name}/config.yaml"
        File out_colormap_fit = "~{name}/colormap_fit.json"
        File out_colormap_project = "~{name}/colormap_project.json"
        File out_fit_labels = "~{name}/data/fit_labels.csv"
        File out_project_labels = "~{name}/data/project_labels.csv"
        # glob, not read_lines: on Google Batch, Cromwell copies back only the
        # outputs it can name before the task runs, and glob is how it names files
        # known only afterwards. A read_lines list was never copied, so the next
        # task's inputs did not exist.
        Array[File] pca_csvs = glob("~{name}/outputs/pca/*.csv")
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
    }
}

task embed {
    input {
        String name
        File config
        File colormap_fit
        File colormap_project
        File fit_labels
        File project_labels
        Array[File] pca_csvs
        Int umap_n_neighbors
        Float umap_min_dist
        String output_dir
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        exec > >(tee embed.log) 2>&1

        mkdir -p exp/data exp/outputs/pca
        cp "~{config}" exp/config.yaml
        cp "~{colormap_fit}" exp/colormap_fit.json
        cp "~{colormap_project}" exp/colormap_project.json
        cp "~{fit_labels}" exp/data/fit_labels.csv
        cp "~{project_labels}" exp/data/project_labels.csv
        cp ~{sep=" " pca_csvs} exp/outputs/pca/

        # 2-D embedding, figures and metrics from the PCA (no genotypes needed).
        manifold-genetics run exp/config.yaml --skip-pca --skip-admixture

        # UMAP and PCA of the same samples and PCs as PHATE (the fit set).
        fit_pca=$(ls exp/outputs/pca/fit_pca_*.csv | head -1)
        manifold-genetics embed --input "${fit_pca}" --method umap \
            --n-neighbors ~{umap_n_neighbors} --min-dist ~{umap_min_dist} \
            --output exp/outputs/embeddings/umap_2d.csv
        cut -d, -f1-3 "${fit_pca}" > exp/outputs/embeddings/pca_2d.csv
        for method in umap pca; do
            manifold-genetics plot --input "exp/outputs/embeddings/${method}_2d.csv" \
                --labels exp/data/fit_labels.csv --colormap exp/colormap_fit.json \
                --output "exp/outputs/figures/embeddings/${method}.png"
        done

        mkdir -p publish/outputs
        for d in embeddings figures metrics; do
            if [ -d "exp/outputs/${d}" ]; then cp -r "exp/outputs/${d}" publish/outputs/; fi
        done
        cp embed.log publish/
        dest="~{output_dir}/~{name}"
        case "${dest}" in
            gs://*) gcloud storage rsync --recursive publish "${dest}" ;;
            *) mkdir -p "${dest}" && cp -R publish/. "${dest}/" ;;
        esac

    >>>

    output {
        # glob, as in subsample_pca. glob does not recurse, so one per figure folder.
        Array[File] embeddings = glob("exp/outputs/embeddings/*.csv")
        Array[File] figures = flatten([
            glob("exp/outputs/figures/embeddings/*.png"),
            glob("exp/outputs/figures/pca/*.png")
        ])
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
    }
}
