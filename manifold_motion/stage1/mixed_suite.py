"""Resource-bounded training driver; one process per seed, variants sequential."""
import argparse
from pathlib import Path
from manifold_motion.stage1.residual_long_rl import run,VARIANTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--root',type=Path,default=Path('reports/manifold_motion/stage1_mixed_v5'))
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS))
    args=p.parse_args()
    original=Path(f'reports/manifold_motion/stage1_ab_response_v3/replicate_{args.seed}/seed_{args.seed}')
    for variant in args.variants:
        print('START',args.seed,variant,flush=True)
        base=original/('no_geometry/imitation.pt' if variant=='no_geometry' else 'imitation.pt')
        run(argparse.Namespace(seed=args.seed,variant=variant,base=base,
             out=args.root/f'seed_{args.seed}'/variant,
             catalog=Path('data/manifold_action_catalog_v1/manifold_action_catalog_v1.npz'),
             iterations=20,contexts=4,candidates=2,ppo_epochs=2))
        print('DONE',args.seed,variant,flush=True)


if __name__=='__main__':main()
