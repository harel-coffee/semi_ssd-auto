from helpers import *


def pseudo_label(pipeline, x_lab, y_lab, x_unlab, y_unlab, threshold=None):
    model = make_pipeline(*pipeline)
    model.fit(x_lab, y_lab)

    pseudo_lab = pd.DataFrame({
        'actual': y_unlab,
        'predict_proba': model.predict_proba(x_unlab)[:, 1]
    })
    if threshold:
        pseudo_lab['predicted'] = (pseudo_lab['predict_proba'] > threshold).astype(int)
        results = {'auc': roc_auc_score(pseudo_lab['actual'], pseudo_lab['predict_proba'])}
    else:
        results = threshold_metrics(pseudo_lab['actual'], pseudo_lab['predict_proba'])
        pseudo_lab['predicted'] = (pseudo_lab['predict_proba'] > results['threshold']).astype(int)

    return pseudo_lab, results


def model_eval(pipeline, x, y, results_path, bucket, rus, random_state, pos_size, run):
    start = datetime.now()

    pos_ratio = np.sum(y) / y.shape[0]
    total = int(pos_size / pos_ratio)
    neg_size = total - pos_size

    x_train, y_train = sample_data(x, y, pos_size, neg_size, random_state + run)

    model = make_pipeline(*pipeline)
    o = cross_validate(model, x_train, y_train, cv=5, scoring=['roc_auc'])
    o['num_instances'] = total
    o['num_pos'] = pos_size
    o['num_neg'] = neg_size
    o['rus'] = rus
    o['run'] = run

    filename = f'{results_path}/rus={rus}/pos_size={pos_size}_rus={rus}.csv'
    if file_exists(filename, bucket):
        out = pd.concat([pd.read_csv(get_s3(filename, bucket)), pd.DataFrame(o)])
    else:
        out = pd.DataFrame(o)
    to_csv_s3(out, filename, bucket)

    logger.info(f"RUN={run} Size={pos_size} RUS={rus} AUC={round(np.mean(o['test_roc_auc']), 4)} Time: {datetime.now() - start}")


def run(bucket, x_path, y_path, results_path, sizes, sparse=False, random_state=42, repeats=5, rus=None, nthreads=10):
    logger.info(f'Creative curve for: {locals()}')

    logger.info(f'Loading data from: {x_path} {y_path}')
    if sparse:
        X = sp.load_npz(get_s3(x_path, bucket))
    else:
        X = np.load(get_s3(x_path, bucket))
    Y = np.load(get_s3(y_path, bucket))
    check_data(X, Y)

    logger.info(f'Sizes to run: {sizes}')

    pipeline = [
        VarianceThreshold(0),
        StandardScaler(with_mean=not sparse),
        SelectKBest(score_func=chi2, k=1000),
        LogisticRegression(random_state=random_state),
    ]
    if rus:
        pipeline = [RatioRandomUnderSampler(rus, random_state=random_state)] + pipeline

    run_args = [(pipeline, X, Y, results_path, bucket, rus, random_state, s, r) for s in sizes for r in range(repeats)]
    random.seed(random_state)
    random.shuffle(run_args)

    with ThreadPool(nthreads) as pool:
        pool.starmap(model_eval, run_args)
