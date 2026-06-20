import pickle
from pprint import pformat

from . import set_logging_debug
from .localization import RetrievalLocalizer, PoseLocalizer, load_valid_images
from .utils.data import Paths, create_argparser, parse_paths, parse_conf
from .utils.io import write_pose_results

experiment = 'pixloc_megadepth'
default_confs = {
    'from_retrieval': {
        'experiment': experiment,
        'features': {},
        'optimizer': {
            'num_iters': 150,
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 3,
            'multiscale': [4, 1],
            'point_selection': 'all',
            'normalize_descriptors': True,
            'average_observations': False,
            'do_pose_approximation': False,
        },
    },
    'from_poses': {
        'experiment': experiment,
        'features': {'preprocessing': {'resize': 1600}},
        'optimizer': {
            'num_iters': 50,
            'pad': 1,
        },
        'refinement': {
            'num_dbs': 5,
            'min_points_opt': 100,
            'point_selection': 'inliers',
            'normalize_descriptors': True,
            'average_observations': True,
            'layer_indices': [0, 1],
        },
    },
}


def main():
    parser = create_argparser('zedx_mini')
    args = parser.parse_args()
    set_logging_debug(args.verbose)

    default_paths = Paths(
        query_images = 'images/' + args.query_set + '/',
        reference_images = 'images/db/',
        reference_sfm = args.running_set + '/sfm_model/',
        query_list = args.running_set + '/' + args.query_set + '/queries_with_intrinsics.txt',
        retrieval_pairs = args.running_set + '/' + args.query_set + '/query_db_pairs.txt',
        results = args.running_set + '/' + args.query_set + '/pixloc_zedx_mini.txt',
    )
    print(f'default paths:\n{pformat(default_paths.asdict())}')
    pose_priors = args.running_set + '/' + args.query_set + '/query_loc.txt_logs.pkl'
    print(f'pose_priors: {pose_priors}')
    paths = parse_paths(args, default_paths)
    conf = parse_conf(args, default_confs)


    if args.from_poses:
        localizer = PoseLocalizer(paths, conf)
    else:
        valid_images = []
        if args.rerun:
            log_txt_file = str(paths.results) + "_logs.pkl.txt"
            valid_images = load_valid_images(log_txt_file)
            print(f"Loaded {len(valid_images)} valid images")
            paths.results = paths.results.with_name(paths.results.stem + '_sup.txt')
        localizer = RetrievalLocalizer(paths, conf, discard_names=valid_images)
    poses, logs = localizer.run_batched(skip=args.skip)

    write_pose_results(poses, paths.results)
    with open(f'{paths.results}_logs.pkl', 'wb') as f:
        pickle.dump(logs, f)


if __name__ == '__main__':
    main()
