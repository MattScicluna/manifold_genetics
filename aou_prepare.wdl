version 1.0

## All of Us, step 1 of 2: prepare a release's cohort (once per release).
##
## Reads the release's microarray genotypes and the CDR's demographics from
## inside the workspace, intersects them with an HGDP+1KGP reference
## (`preprocess --preset harmonise`), and writes the prepared cohort directory
## to <output_dir>/prepared/. Step 2 (analyse.wdl) starts from there.
##
## Acquire and preprocess run on one machine: the release's arrays.bed is
## ~230 GB for v9, and only the prepared cohort (~12 GB) leaves it.
##
## The reference is a PLINK set of HGDP+1KGP samples with the population in the
## FID (`<Pop>` or `forReference<Pop>`). For now it is an internal extract from
## the published analysis; see issue #172.

workflow aou_prepare {
    input {
        # Where the results go, e.g. gs://<bucket>/aou_v9 (writes <output_dir>/prepared/).
        String output_dir

        # The workspace's CDR, as $WORKSPACE_CDR in any app (e.g. <project>.C2025Q4R6).
        String workspace_cdr
        # The release's storage root, as $CDR_STORAGE_PATH in any app.
        String cdr_storage_path = "gs://vwb-aou-datasets-controlled/v9"

        File reference_bed
        File reference_bim
        File reference_fam

        String docker = "us-central1-docker.pkg.dev/all-of-us-rw-prod/aou-rw-gar-remote-repo-docker-prod/mattscicluna/manifold-genetics:verily-workflow"
        Int cpu = 32
        Int memory_gb = 128
        Int disk_gb = 1000
    }

    call prepare {
        input:
            output_dir = output_dir,
            workspace_cdr = workspace_cdr,
            cdr_storage_path = cdr_storage_path,
            reference_bed = reference_bed,
            reference_bim = reference_bim,
            reference_fam = reference_fam,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb
    }

    output {
        File config = prepare.config
        File log = prepare.log
    }
}

task prepare {
    input {
        String output_dir
        String workspace_cdr
        String cdr_storage_path
        File reference_bed
        File reference_bim
        File reference_fam
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        exec > >(tee prepare.log) 2>&1

        # What `load-env.sh` provides in an app. The billing project is the
        # machine's own project, read from the metadata server.
        GOOGLE_PROJECT="$(curl -fsS -H 'Metadata-Flavor: Google' \
            http://metadata.google.internal/computeMetadata/v1/project/project-id)"
        export GOOGLE_PROJECT
        export WORKSPACE_CDR="~{workspace_cdr}"
        export CDR_STORAGE_PATH="~{cdr_storage_path}"
        echo "project ${GOOGLE_PROJECT}; CDR ${WORKSPACE_CDR}; release ${CDR_STORAGE_PATH}"

        # The reference cohort: population from the FID, sample IDs from the IID.
        mkdir -p ref_raw
        ln -s "~{reference_bed}" ref_raw/reference.bed
        ln -s "~{reference_bim}" ref_raw/reference.bim
        ln -s "~{reference_fam}" ref_raw/reference.fam
        awk 'BEGIN { print "sample_id,Population" }
             { p = $1; sub(/^forReference/, "", p); print $2 "," p }' \
            ref_raw/reference.fam > ref_raw/labels.csv
        manifold-genetics acquire custom \
            --fit-plink "$PWD/ref_raw/reference" --labels "$PWD/ref_raw/labels.csv" --out ref

        manifold-genetics acquire aou --out aou

        manifold-genetics preprocess ref/config.yaml aou/config.yaml \
            --preset harmonise --skip-wrayner --threads ~{cpu} \
            --memory $(( ~{memory_gb} * 1024 * 3 / 4 )) --cleanup --out prepared

        # Publish the prepared cohort: real files, not the links to the download.
        mkdir -p publish/data
        cp prepared/config.yaml prepared/colormap_fit.json prepared/colormap_project.json publish/
        cp -L prepared/data/*_subset.bed prepared/data/*_subset.bim prepared/data/*_subset.fam \
            prepared/data/*_labels.csv publish/data/
        gcloud storage rsync --recursive publish "~{output_dir}/prepared"
        gcloud storage cp prepare.log "~{output_dir}/prepared/prepare.log"
        cp prepared/config.yaml config.yaml
    >>>

    output {
        File config = "config.yaml"
        File log = "prepare.log"
    }

    runtime {
        docker: docker
        cpu: cpu
        memory: "~{memory_gb} GB"
        disks: "local-disk ~{disk_gb} SSD"
        preemptible: 0
    }
}
