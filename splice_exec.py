import learning_curve

# using r4.4xlarge
learning_curve.run(
    bucket='fau-bigdata',
    x_path='semi_ssd/data/splice/splice_X.npz',
    y_path='semi_ssd/data/splice/splice_Y.npy',
    results_path='semi_ssd/results/splice/actual',
    sizes=list(range(10, 100, 10)) + list(range(100, 7000, 100)),
    sparse=True,
    random_state=42,
)

learning_curve.run(
    bucket='fau-bigdata',
    x_path='semi_ssd/data/splice/splice_X.npz',
    y_path='semi_ssd/data/splice/splice_Y.npy',
    results_path='semi_ssd/results/splice/actual',
    sizes=list(range(10, 100, 10)) + list(range(100, 7000, 100)),
    sparse=True,
    random_state=42,
    rus=0.5,
)

learning_curve.run(
    bucket='fau-bigdata',
    x_path='semi_ssd/data/splice/splice_X.npz',
    y_path='semi_ssd/data/splice/splice_Y.npy',
    results_path='semi_ssd/results/splice/actual',
    sizes=list(range(10, 100, 10)) + list(range(100, 7000, 100)),
    sparse=True,
    random_state=42,
    rus=0.25,
)
