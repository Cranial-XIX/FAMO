cp config.py              ../mtrl/utils/
cp metaworld.yaml         ../config/experiment/
cp mtrl.yaml              ../config/experiment/

# copy cagrad
cp cagrad.py              ../mtrl/agent/
cp cagrad_state_sac.yaml  ../config/agent/
# copy nashmtl
cp nashmtl.py             ../mtrl/agent/
cp nashmtl_state_sac.yaml ../config/agent/
# copy uw
cp uw.py                  ../mtrl/agent/
cp uw_state_sac.yaml      ../config/agent/
# copy famo
cp famo.py                ../mtrl/agent/
cp famo_state_sac.yaml    ../config/agent/

cp run.sh                 ../
