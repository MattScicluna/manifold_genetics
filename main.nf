#!/usr/bin/env nextflow
/*
 * manifold-genetics as a Nextflow workflow (Verily Workbench, Google Batch, or local).
 *
 *   nextflow run main.nf -profile full         --fit_plink ... --labels ... --colormap ...
 *   nextflow run main.nf -profile no_admixture --fit_plink ... --labels ... --colormap ...
 *
 * One workflow, two profiles: `full` runs everything; `no_admixture` sets
 * params.run_admixture = false, which passes --skip-admixture and so leaves out
 * the admixture step, its plots and its metrics. PCA, embedding, the standard
 * plots and the non-admixture metrics run in both.
 *
 * Today the work is one process wrapping `manifold-genetics pipeline`. It sits
 * inside the MANIFOLD_GENETICS sub-workflow, whose inputs and outputs are the
 * ones the steps exchange, so PCA, admixture, embedding, plotting and metrics
 * can become separate processes later without changing this entry point.
 */

nextflow.enable.dsl = 2

/* A PLINK prefix as its three files; fail early, naming what is missing. */
def plinkTriple(String prefix, String what) {
    def files = ['bed', 'bim', 'fam'].collect { file("${prefix}.${it}") }
    def missing = files.findAll { !it.exists() }
    if (missing) {
        error "${what}: ${missing*.toString().join(', ')} not found (give the prefix, without .bed/.bim/.fam)"
    }
    return files
}

/* An optional input: the file, or a placeholder the script recognises as absent. */
def optionalFile(value, String name) {
    value ? file(value, checkIfExists: true) : file("${projectDir}/assets/nextflow/NO_FILE_${name}")
}

/* Staged under stageAs 'dir/*', a file's name includes that dir, so look for the marker anywhere. */
def present(f) { !f.toString().contains('NO_FILE_') }

/* The prefix of a staged PLINK triple: the .bed's name without its extension. */
def plinkPrefix(files) { files.find { it.name.endsWith('.bed') }.baseName }

process MANIFOLD_GENETICS_PIPELINE {
    tag "${params.run_admixture ? 'full' : 'no_admixture'}"
    label 'pipeline'
    publishDir params.outdir, mode: 'copy'

    input:
    path fit_plink,       stageAs: 'fit/*'
    path project_plink,   stageAs: 'project/*'
    path labels,          stageAs: 'labels/*'
    path colormap,        stageAs: 'colormap/*'
    path fit_labels,      stageAs: 'fit_labels/*'
    path project_labels,  stageAs: 'project_labels/*'
    path fit_colormap,    stageAs: 'fit_colormap/*'
    path project_colormap, stageAs: 'project_colormap/*'
    path geographic,      stageAs: 'geographic/*'

    output:
    path 'results/**', emit: results

    script:
    def args = []
    args << "--fit-plink fit/${plinkPrefix(fit_plink)}"
    args << "--project-plink project/${plinkPrefix(project_plink)}"
    if (present(labels))           args << "--labels ${labels}"
    if (present(colormap))         args << "--colormap ${colormap}"
    if (present(fit_labels))       args << "--fit-labels ${fit_labels}"
    if (present(project_labels))   args << "--project-labels ${project_labels}"
    if (present(fit_colormap))     args << "--fit-colormap ${fit_colormap}"
    if (present(project_colormap)) args << "--project-colormap ${project_colormap}"
    if (present(geographic))       args << "--geographic ${geographic}"
    args << "--output results"
    args << "--threads ${task.cpus}"
    args << "--n-pcs ${params.n_pcs}"
    args << "--pca-backend ${params.pca_backend}"
    args << "--embedding ${params.embedding}"
    args << "--embedding-input ${params.embedding_input}"
    args << "--n-components ${params.n_components}"
    if (params.knn != null)         args << "--knn ${params.knn}"
    if (params.t != null)           args << "--t ${params.t}"
    if (params.n_landmark != null)  args << "--n-landmark ${params.n_landmark}"
    if (params.random_landmarking)  args << "--random-landmarking"
    if (params.run_admixture) {
        args << "--k-min ${params.k_min} --k-max ${params.k_max}"
        if (params.gpus) args << "--num-gpus ${params.gpus}"
    } else {
        args << "--skip-admixture"
    }
    if (params.extra_args) args << params.extra_args
    """
    manifold-genetics pipeline ${args.join(' ')}
    """

    stub:
    """
    mkdir -p results/pca results/embeddings results/figures results/metrics
    touch results/pca/fit_pca.csv results/embeddings/embedding.csv
    ${params.run_admixture ? 'mkdir -p results/admixture && touch results/admixture/Q.txt' : ''}
    """
}

/*
 * The steps' contract. Split later into PCA -> (ADMIXTURE) -> EMBED -> PLOT -> METRICS,
 * each taking the previous step's outputs, without touching the entry workflow.
 */
workflow MANIFOLD_GENETICS {
    take:
    fit_plink
    project_plink
    labels
    colormap
    fit_labels
    project_labels
    fit_colormap
    project_colormap
    geographic

    main:
    MANIFOLD_GENETICS_PIPELINE(
        fit_plink, project_plink, labels, colormap,
        fit_labels, project_labels, fit_colormap, project_colormap, geographic
    )

    emit:
    results = MANIFOLD_GENETICS_PIPELINE.out.results
}

workflow {
    if (!params.fit_plink) {
        error "Set --fit_plink (a PLINK prefix). See docs/nextflow.md."
    }
    if (!params.labels && !(params.fit_labels && params.project_labels)) {
        error "Set --labels, or both --fit_labels and --project_labels."
    }
    if (!params.colormap && !(params.fit_colormap && params.project_colormap)) {
        error "Set --colormap, or both --fit_colormap and --project_colormap."
    }

    log.info """\
        manifold-genetics ${params.run_admixture ? 'full' : 'no_admixture'}
        fit     : ${params.fit_plink}
        project : ${params.project_plink ?: params.fit_plink}
        outdir  : ${params.outdir}
        """.stripIndent()

    MANIFOLD_GENETICS(
        Channel.value(plinkTriple(params.fit_plink, 'fit_plink')),
        Channel.value(plinkTriple(params.project_plink ?: params.fit_plink, 'project_plink')),
        Channel.value(optionalFile(params.labels, 'labels')),
        Channel.value(optionalFile(params.colormap, 'colormap')),
        Channel.value(optionalFile(params.fit_labels, 'fit_labels')),
        Channel.value(optionalFile(params.project_labels, 'project_labels')),
        Channel.value(optionalFile(params.fit_colormap, 'fit_colormap')),
        Channel.value(optionalFile(params.project_colormap, 'project_colormap')),
        Channel.value(optionalFile(params.geographic, 'geographic')),
    )
}
