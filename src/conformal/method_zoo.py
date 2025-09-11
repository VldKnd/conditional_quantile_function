from conformal.classes.conformalizers import OTCPGlobalPredictor, OTCPLocalPredictor, SplitConformalPredictor, EllipsoidalLocal
from conformal.classes.method_desc import ConformalMethodDescription


section5 = [
    ConformalMethodDescription(
        name="PB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{pb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Rank",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
    ),
    ConformalMethodDescription(
        name="RPB",
        name_mathtext=r"$\mathcal{C}^{\mathrm{rpb}}$",
        base_model_name="CVQRegressor",
        score_name="MK Quantile",
        class_name="OTCPGlobalPredictor",
        cls=OTCPGlobalPredictor
    ),
    ConformalMethodDescription(
        name="HPD",
        name_mathtext=r"$\mathcal{C}^{\mathrm{HPD}}$",
        base_model_name="CVQRegressor",
        score_name="Log Density",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
        kwargs=dict(lower_is_better=False)
    ),

]


cpflow_based = [
    ConformalMethodDescription(
        name="PB (CPFlow)",
        name_mathtext=r"$\mathcal{C}^{\mathrm{pb}}$ (CPFlow)",
        base_model_name="CPFlowRegressor",
        score_name="MK Rank",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
    ),
    ConformalMethodDescription(
        name="RPB (CPFlow)",
        name_mathtext=r"$\mathcal{C}^{\mathrm{rpb}}$ (CPFlow)",
        base_model_name="CPFlowRegressor",
        score_name="MK Quantile",
        class_name="OTCPGlobalPredictor",
        cls=OTCPGlobalPredictor
    ),
    ConformalMethodDescription(
        name="HPD (CPFlow)",
        name_mathtext=r"$\mathcal{C}^{\mathrm{HPD}}$ (CPFlow)",
        base_model_name="CPFlowRegressor",
        score_name="Log Density",
        class_name="SplitConformalPredictor",
        cls=SplitConformalPredictor,
        kwargs=dict(lower_is_better=False)
    ),

]


baselines = [
    ConformalMethodDescription(
        name="OT-CP-Global",
        name_mathtext=r"$\mathrm{OT}$-$\mathrm{CP}$",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="OTCPGlobalPredictor",
        cls=OTCPGlobalPredictor
    ),
    ConformalMethodDescription(
        name="OT-CP-Local",
        name_mathtext=r"$\mathrm{OT}$-$\mathrm{CP}$+",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="OTCPLocalPredictor",
        cls=OTCPLocalPredictor
    ),
    ConformalMethodDescription(
        name="Ell-Local",
        name_mathtext=r"$\mathrm{ELL}$-$\mathrm{local}$+",
        base_model_name="RandomForest",
        score_name="Signed Error",
        class_name="EllipsoidalLocal",
        cls=EllipsoidalLocal
    ),

]