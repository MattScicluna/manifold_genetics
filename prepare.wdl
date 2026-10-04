version 1.0

## Any biobank, step 1 of 2: prepare a cohort you already have as PLINK files
## (UK Biobank, or your own). The All of Us version is aou_prepare.wdl, which
## fetches the release itself; everything after that fetch is the same.
##
## Intersects the cohort with an HGDP+1KGP reference (`preprocess --preset
## harmonise`, the flags aou_prepare.wdl uses) and writes the prepared cohort
## directory to <output_dir>/prepared/. Step 2 (analyse.wdl) starts from there.
##
## The reference is the one part that differs between biobanks: the same
## HGDP+1KGP samples, restricted to the positions of this cohort's array, so
## the intersection keeps as many SNPs as the array allows. Its population is
## read from reference_labels when given, otherwise from the FID (`<Pop>` or
## `forReference<Pop>`), as in aou_prepare.wdl.
##
## Colours: labels with published colours (HGDP+1KGP, UK Biobank, All of Us)
## get them; any other value gets a generated one. cohort_colormap and
## reference_colormap override them.
##
## The cohort's chromosomes are expected as `chr1`, as in the All of Us and
## UK Biobank GRCh38 releases; the reference's may be either (see
## reference_has_chr_prefix).

workflow prepare {
    input {
        # Where the results go (writes <output_dir>/prepared/): gs://... or a local path.
        String output_dir
        File cohort_bed
        File cohort_bim
        File cohort_fam
        # sample_id plus the columns to colour by (e.g. self_described_ancestry).
        File cohort_labels
        File reference_bed
        File reference_bim
        File reference_fam
        # sample_id plus columns (e.g. Population); default: population from the FID.
        File? reference_labels
        Boolean reference_has_chr_prefix = false
        # Optional colormap JSONs (column -> value -> colour) that take precedence.
        File? cohort_colormap
        File? reference_colormap
        String docker = "mattscicluna/manifold-genetics:main"
        Int cpu = 32
        Int memory_gb = 128
        Int disk_gb = 200
    }

    call prepare_cohort {
        input:
            output_dir = output_dir,
            cohort_bed = cohort_bed,
            cohort_bim = cohort_bim,
            cohort_fam = cohort_fam,
            cohort_labels = cohort_labels,
            reference_bed = reference_bed,
            reference_bim = reference_bim,
            reference_fam = reference_fam,
            reference_labels = reference_labels,
            reference_has_chr_prefix = reference_has_chr_prefix,
            cohort_colormap = cohort_colormap,
            reference_colormap = reference_colormap,
            docker = docker,
            cpu = cpu,
            memory_gb = memory_gb,
            disk_gb = disk_gb
    }

    output {
        File config = prepare_cohort.config
        File log = prepare_cohort.log
    }
}

task prepare_cohort {
    input {
        String output_dir
        File cohort_bed
        File cohort_bim
        File cohort_fam
        File cohort_labels
        File reference_bed
        File reference_bim
        File reference_fam
        File? reference_labels
        Boolean reference_has_chr_prefix
        File? cohort_colormap
        File? reference_colormap
        String docker
        Int cpu
        Int memory_gb
        Int disk_gb
    }

    command <<<
        set -euo pipefail
        exec > >(tee prepare.log) 2>&1

        # Each PLINK set under one prefix, linked rather than copied.
        mkdir -p ref_raw cohort_raw
        ln -s "~{reference_bed}" ref_raw/reference.bed
        ln -s "~{reference_bim}" ref_raw/reference.bim
        ln -s "~{reference_fam}" ref_raw/reference.fam
        ln -s "~{cohort_bed}" cohort_raw/cohort.bed
        ln -s "~{cohort_bim}" cohort_raw/cohort.bim
        ln -s "~{cohort_fam}" cohort_raw/cohort.fam

        ref_labels="~{default='' reference_labels}"
        if [ -z "${ref_labels}" ]; then
            # Population from the FID, sample IDs from the IID.
            ref_labels="$PWD/ref_raw/labels.csv"
            awk 'BEGIN { print "sample_id,Population" }
                 { p = $1; sub(/^forReference/, "", p); print $2 "," p }' \
                ref_raw/reference.fam > "${ref_labels}"
        fi

        manifold-genetics acquire custom \
            --fit-plink "$PWD/ref_raw/reference" --labels "${ref_labels}" \
            ~{"--colormap " + reference_colormap} --out ref
        manifold-genetics acquire custom \
            --fit-plink "$PWD/cohort_raw/cohort" --labels "~{cohort_labels}" \
            ~{"--colormap " + cohort_colormap} --out cohort
        manifold-genetics preprocess ref/config.yaml cohort/config.yaml \
            --preset harmonise --skip-wrayner --threads ~{cpu} \
            ~{if reference_has_chr_prefix then "--fit-has-chr-prefix" else ""} \
            --memory $(( ~{memory_gb} * 1024 * 3 / 4 )) --cleanup --out prepared

        # Publish the prepared cohort: real files, not the links to the inputs.
        mkdir -p publish/data
        cp prepared/config.yaml prepared/colormap_fit.json prepared/colormap_project.json publish/
        cp -L prepared/data/*_subset.bed prepared/data/*_subset.bim prepared/data/*_subset.fam \
            prepared/data/*_labels.csv publish/data/
        cp prepare.log publish/
        dest="~{output_dir}/prepared"
        case "${dest}" in
            gs://*) gcloud storage rsync --recursive publish "${dest}" ;;
            *) mkdir -p "${dest}" && cp -R publish/. "${dest}/" ;;
        esac
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
