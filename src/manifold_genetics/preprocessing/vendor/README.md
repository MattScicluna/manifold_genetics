# Vendored: HRC-1000G-check-bim v4.3.0

`HRC-1000G-check-bim-v4.3.0.zip` is William Rayner's HRC/1000G/TOPMed pre-imputation
checker, as published at
`https://www.chg.ox.ac.uk/~wrayner/tools/HRC-1000G-check-bim-v4.3.0.zip`.
That URL now returns 404; this copy is from the Internet Archive
(`web.archive.org/web/20240415210219/...`, identical to the 2022 snapshot of the
older `www.well.ox.ac.uk` URL), unmodified:

    sha256  c9bc5a02f6209ffcf5c0e09e72f6f0f06dd6f2ed97569a89543edbd3d1077549

MIT License, copyright (c) 2018 William Rayner — `LICENSE-HRC-1000G-check-bim.txt`
(also inside the zip). `preprocess --preset harmonise` runs it; the installer
unzips it and comments out the line that writes a VCF, as the shell always has.
