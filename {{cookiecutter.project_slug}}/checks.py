from art.step.checks import Check, CheckResult, ResultOfCheck
from art.step.step import Step
from art.step.step_savers import MatplotLibSaver


class CheckClassImagesExist(Check):
    def check(self, step: Step) -> ResultOfCheck:
        for class_name in step.get_latest_run()["class_names"]:
            image_path = step.get_class_image_path(class_name)
            if not MatplotLibSaver().exists(step.get_step_id(), step.name, image_path):
                return ResultOfCheck(
                    is_positive=False,
                    error=f"Image for class: {class_name} does not exist. it should have been here: {MatplotLibSaver().get_path(step.get_step_id(), step.name, image_path)}",
                )
        return ResultOfCheck(is_positive=True)


class CheckLenClassNamesEqualToNumClasses(CheckResult):
    def _check_method(self, result) -> ResultOfCheck:
        if len(result["class_names"]) != result["number_of_classes"]:
            return ResultOfCheck(
                is_positive=False,
                error="Number of class names is different than number of classes",
            )
        return ResultOfCheck(is_positive=True)
