class ExperimentMetrics:
    def __init__(self):
        self.exp_dir = ""
        self.stage = ""
        self.seed = ""
        self.backbone = ""
        self.unfrozen_layers = ""
        self.source_dataset = ""
        self.target_dataset = ""
        self.batch_size = ""
        self.lr = ""
        self.epochs = ""
        self.es_patience = ""
        self.reduce_lr_patience = ""
        self.n_clusters = ""
        self.clustering = ""
        self.undersampling = ""
        self.ae_arch = ""
        self.align_weight = ""
        self.kl_weight = ""
        self.reconstruction_weight = ""
        self.source_loss = ""
        self.source_acc = ""
        self.source_pre = ""
        self.source_rec = ""
        self.source_f1 = ""
        self.target_loss = ""
        self.target_acc = ""
        self.target_pre = ""
        self.target_rec = ""
        self.target_f1 = ""
        self.aligned_loss = ""
        self.aligned_acc = ""
        self.aligned_pre = ""
        self.aligned_rec = ""
        self.aligned_f1 = ""

    def to_json(self):
        return {
            "experiment": self.exp_dir,
            "stage": self.stage,
            "seed": self.seed,
            "backbone": self.backbone,
            "unfrozen_layers": self.unfrozen_layers,
            "source_dataset": self.source_dataset,
            "target_dataset":  self.target_dataset,
            "batch_size":  self.batch_size,
            "lr":  self.lr,
            "epochs":  self.epochs,
            "es_patience":  self.es_patience,
            "reduce_lr_patience":  self.reduce_lr_patience,
            "n_clusters":  self.n_clusters,
            "clustering":  self.clustering,
            "undersampling":  self.undersampling,
            "ae_arch":  self.ae_arch,
            "align_weight":  self.align_weight,
            "kl_weight":  self.kl_weight,
            "reconstruction_weight":  self.reconstruction_weight,
            "source_loss":  self.source_loss,
            "source_acc":  self.source_acc,
            "source_pre":  self.source_pre,
            "source_rec":  self.source_rec,
            "source_f1":  self.source_f1,
            "target_loss":  self.target_loss,
            "target_acc":  self.target_acc,
            "target_pre":  self.target_pre,
            "target_rec":  self.target_rec,
            "target_f1":  self.target_f1,
            "aligned_loss":  self.aligned_loss,
            "aligned_acc":  self.aligned_acc,
            "aligned_pre":  self.aligned_pre,
            "aligned_rec":  self.aligned_rec,
            "aligned_f1":  self.aligned_f1
        }