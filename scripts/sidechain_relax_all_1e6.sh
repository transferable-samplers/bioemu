#!/bin/bash
#SBATCH --job-name=bioemu_relax_1e6
#SBATCH -o watch_folder/%x_%j.out     # output file (%j expands to jobID)
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --partition=main-cpu,long-cpu
#SBATCH -t 12:00:00
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --array=0-91
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

sequences=(
    AA
    CE
    CL
    DG
    DI
    FK
    HL
    HM
    IK
    IM
    KG
    LE
    MQ
    NA
    NC
    PG
    PY
    QR
    RL
    RT
    SS
    TD
    VF
    VS
    WA
    WH
    WQ
    WS
    YC
    YQ
    ARIP
    CCVH
    CIPQ
    DEMT
    DMTL
    EHQW
    FESD
    FYYY
    GCDE
    GDTI
    GGRS
    HEAV
    HQVS
    HYGW
    ITYL
    KKAP
    KLLR
    KRWN
    NCFG
    NEVI
    PQIF
    QAKR
    QWNL
    RLMM
    SHKS
    SVND
    TAPF
    TMWC
    VPFY
    WNMA
    ANKSMIEA
    CGSWHKQR
    CLCCGQWN
    DDRDTEQT
    DGVAHALS
    EKYYWMQT
    FWRVDHDM
    GNDLVTVI
    HWHSLICK
    IDHRQLKW
    IFGWVYTG
    ISKCKNGE
    KRRGFFLE
    MAPQTIAT
    MRDPVLFA
    MWNSTEMI
    MYGRNCYM
    NHQYGSDP
    NKEKFFQH
    NPCLCYML
    PGESTAES
    PLFHVMYV
    PPWRECNN
    PYIRNCVE
    SPHKMRLC
    SQQKVAFE
    VWIPVIDT
    WDLIQFRQ
    WTYAFAHS
    YFPHAGYT
    YQNPDGSQA
    GYDPETGTWG
)

# Index from the Slurm array
idx="${SLURM_ARRAY_TASK_ID}"
sequence="${sequences[$idx]}"

python -m bioemu.sidechain_relax \
    --input-dir "/network/scratch/t/tanc/bioemu_100/$sequence" \
    --output-subdir "init_equil" \
    --energy-eval-budget 1_000_000 \
    --md-protocol md_equil \
    --reference-pdb-path="/network/scratch/t/tanc/transferable-samplers/many-peptides-md/pdbs/test/${sequence}.pdb"
