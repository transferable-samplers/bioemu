#!/bin/bash
#SBATCH --job-name=bioemu_relax
#SBATCH --get-user-env                # retrieve the users login environment
#SBATCH --partition=main-cpu,long-cpu
#SBATCH -t 12:00:00
#SBATCH -c 1
#SBATCH --mem=8G
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
    sequence $sequence \
    output_dir "/network/scratch/t/tanc/bioemu-1e4/$sequence" \
    energy_eval_budget 10_000 # if you increase this enough it may be worth adding simtime_ns - think though about how to best sweep, maybe just take 100 points and then run MD? 10?


    # reduced to 0.01ps for each step size as otherwise
    # ~200,000 energy evaluations used probably ok as the sequences are much smaller
    # than this was originally intended for