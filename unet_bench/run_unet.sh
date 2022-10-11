time $PYTHON unet.py --hpu --use_lazy_mode --run-name habana-worker-1-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-1-d.pt
time mpirun -n 2 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-2-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-2-d.pt --world-size 2 --num-workers 2
time mpirun -n 4 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-4-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-4-d.pt --world-size 4 --num-workers 4
time mpirun -n 8 --rank-by core $PYTHON unet.py --hpu --use_lazy_mode --distributed --run-name habana-worker-8-duped --epochs 5 --cache-path kaggle_duped_cache --weights-file h-w-8-d.pt --world-size 8 --num-workers 8
