version 1.0

## 3-D figures for experiments analyse.wdl has run: PHATE, UMAP and the first
## three PCs of each experiment's fit set, each as a rotatable HTML and a
## rotating MP4. Reads the PCA analyse.wdl published to <output_dir>/<name>/,
## so it needs no genotypes and does not redo the PCA; writes to the same
## folder (outputs/embeddings/*_3d.csv, outputs/figures/embeddings_3d/).

workflow analyse_3d {
    input {
        # analyse.wdl's output_dir, e.g. gs://<bucket>/aou_v9
        String output_dir
        # analyse.wdl's experiment names.
        Array[String] experiments = ["balanced", "geosketch_90k"]
        Int n_pcs = 20

        # PHATE: the subsample preset's settings.
        Int knn = 500
        Int t = 50
        Int n_landmark = 10000
        Int umap_n_neighbors = 15
        Float umap_min_dist = 0.5

        String docker = "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/mattscicluna/manifold-genetics:verily-workflow"
        Int cpu = 8
        Int memory_gb = 64
        Int disk_gb = 50
    }

    scatter (name in experiments) {
        call embed_3d {
            input:
                name = name,
                fit_pca = output_dir + "/" + name + "/outputs/pca/fit_pca_" + n_pcs + ".csv",
                fit_labels = output_dir + "/" + name + "/data/fit_labels.csv",
                colormap_fit = output_dir + "/" + name + "/colormap_fit.json",
                knn = knn,
                t = t,
                n_landmark = n_landmark,
                umap_n_neighbors = umap_n_neighbors,
                umap_min_dist = umap_min_dist,
                output_dir = output_dir,
                docker = docker,
                cpu = cpu,
                memory_gb = memory_gb,
                disk_gb = disk_gb
        }
    }

    output {
        Array[File] embeddings = flatten(embed_3d.embeddings)
        Array[File] figures = flatten(embed_3d.figures)
    }
}

task embed_3d {
    input {
        String name
        File fit_pca
        File fit_labels
        File colormap_fit
        Int knn
        Int t
        Int n_landmark
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
        exec > >(tee embed_3d.log) 2>&1

        mkdir -p exp/outputs/embeddings exp/outputs/figures/embeddings_3d
        landmarks=""
        if [ ~{n_landmark} -gt 0 ]; then landmarks="--n-landmark ~{n_landmark} --random-landmarking"; fi
        manifold-genetics embed --input "~{fit_pca}" --method phate \
            --knn ~{knn} --t ~{t} --n-components 3 ${landmarks} \
            --output exp/outputs/embeddings/phate_3d.csv
        manifold-genetics embed --input "~{fit_pca}" --method umap \
            --n-neighbors ~{umap_n_neighbors} --min-dist ~{umap_min_dist} --n-components 3 \
            --output exp/outputs/embeddings/umap_3d.csv
        cut -d, -f1-4 "~{fit_pca}" > exp/outputs/embeddings/pca_3d.csv

        for method in phate umap pca; do
            manifold-genetics plot-3d --input "exp/outputs/embeddings/${method}_3d.csv" \
                --labels "~{fit_labels}" --colormap "~{colormap_fit}" \
                --output exp/outputs/figures/embeddings_3d --prefix "${method}_3d" \
                --max-points 0 --no-hover-ids
        done

        mkdir -p publish
        cp -r exp/outputs publish/
        cp embed_3d.log publish/
        dest="~{output_dir}/~{name}"
        case "${dest}" in
            gs://*) gcloud storage rsync --recursive publish "${dest}" ;;
            *) mkdir -p "${dest}" && cp -R publish/. "${dest}/" ;;
        esac

    >>>

    output {
        # glob, as in analyse.wdl: Google Batch copies back only named outputs.
        Array[File] embeddings = glob("exp/outputs/embeddings/*.csv")
        Array[File] figures = glob("exp/outputs/figures/embeddings_3d/*")
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
    }
}
