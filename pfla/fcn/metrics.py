from linear import Linear as ln

class Metrics:
    """Compute various metrics based on landmarks

    Parameters
    ----------
    LDMK: numpy array
        Array containing the landmarks for the detected face

    Returns
    -------
    metrics: numpy array
        Numpy array of the metrics computed
    """

    def __init__(self, LDMK):
        self.ldmk = LDMK

    def compute_metrics(self):
        fh = ln(self.ldmk[27][0], self.ldmk[27][1],
                self.ldmk[8][0], self.ldmk[8][1])
        fw = ln(self.ldmk[1][0], self.ldmk[1][1],
                self.ldmk[15][0], self.ldmk[15][1])
        lf = ln(self.ldmk[33][0], self.ldmk[33][1],
                self.ldmk[8][0], self.ldmk[8][1])
        mw = ln(self.ldmk[4][0], self.ldmk[4][1],
                self.ldmk[12][0], self.ldmk[12][1])

        facial_index = fh.euc_dist() / fw.euc_dist()
        lf_fh_index = lf.euc_dist() / fh.euc_dist()
        mw_fw_index = mw.euc_dist() / fw.euc_dist()
        mw_fh_index = mw.euc_dist() / fh.euc_dist()

        metrics = [facial_index, lf_fh_index, mw_fw_index, mw_fh_index]

        return metrics
