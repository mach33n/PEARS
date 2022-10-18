class MetricsBin:
    def __init__(self):
        # Insert any necessary variables here.
        # You may want to add parameters for things like 
        # the global dataloader object.
        pass

    def F1(self, preds, truths):
        # Given a list of prediction values(possessing -1,0,1),
        # compute the F1 score with respect to the ground truth
        # values provided.
        pass

    def avgVariance(self, preds, truths):
        # Given a list of prediction values(possessing -1,0,1),
        # compute the average variance score with respect to the 
        # ground truth values. This may require you add a parameter 
        # for original input corresponding to each prediction.
        pass

    def avgDiff(self, preds, truths):
        # Given a list of prediction values(possessing -1,0,1),
        # compute the average difference score with respect to the 
        # ground truth values. This may require you add a parameter 
        # for original input corresponding to each prediction.
        pass