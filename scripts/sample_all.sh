#!/bin/bash
#SBATCH --job-name=bioemu_sample
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --partition=long
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --array=0-91
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# YOU MAY NEED TO export BIOEMU_COLABFOLD_DIR=/home/mila/t/tanc/bioemu/colabfold

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

python -m bioemu.sample \
    --sequence $sequence \
    --num_samples 10_000 \
    --output_dir "/network/scratch/t/tanc/bioemu-1e4/$sequence" \
    --filter_samples=False
