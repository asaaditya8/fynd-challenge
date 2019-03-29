model.load_weights()

    scores = model.evaluate_generator(gen_dict['train'], steps=total_samples // BATCH_SIZE,
                                      batch_size=BATCH_SIZE, callbacks=[])

    for score, metric_name in zip(scores, model.metrics_names):
        print("%s : %0.4f" % (metric_name, score))