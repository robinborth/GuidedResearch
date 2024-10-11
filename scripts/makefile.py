from pathlib import Path

SUFFIX = "abl"


def float_to_scientific(values):
    return [f"{v:.1E}".replace(".", "-") for v in values]


def generate_banner(group_names: list[str]):
    b = ""
    b += "##########################################################################\n"
    b += f"# make all -f Makefile.{SUFFIX}\n"
    for i, group in enumerate(group_names):
        b += f"# GROUP{i}: make {group} -f Makefile.{SUFFIX}\n"
    b += "##########################################################################\n"
    return b


def generate_group_banner(group_name: str = "GROUP"):
    b = ""
    b += "##########################################################################\n"
    b += f"# {group_name}\n"
    b += "##########################################################################\n"
    b += "\n"
    return b


def generate_make_command(template_generator, value, group_name, prefix):
    task_name = f"{group_name}__{prefix}"
    make_command = f"{task_name}:"
    template = template_generator.format(
        task_name=task_name, value=value, group_name=group_name
    )
    template = make_command + template
    return task_name, template


def build_group(template_generator, values, prefixs, group_name: str = "GROUP"):
    templates = []
    task_names = []
    for value, prefix in zip(values, prefixs):
        task_name, template = generate_make_command(
            template_generator=template_generator,
            value=value,
            group_name=group_name,
            prefix=prefix,
        )
        t = "\n\t".join(
            [li.strip() for li in template.split("\n") if li and li.strip()]
        )
        task_names.append(task_name)
        templates.append(t)

    temp = generate_group_banner(group_name)
    tasks = " ".join(task_names)
    temp += f"\n{group_name}: {tasks}\n\n"
    temp += "\n\n".join(templates)
    temp += "\n"

    return temp, group_name, tasks


def build_makefile(groups):
    templates, group_names, task_names = [], [], []
    for group in groups:
        temp, names, task = group
        templates.append(temp)
        group_names.append(names)
        task_names.append(task)
    g = generate_banner(group_names)
    g += "\n"
    tasks = " ".join(task_names)
    g += f".PHONY: all {tasks}\n"
    g += f"all: {tasks}\n\n"
    g += "\n\n".join(templates)
    g += "\n\n"
    return g


def main():
    groups = []

    values = [0.7, 0.5, 0.3, 0.1]
    prefixs = float_to_scientific(values)
    group_name = "train_step_size"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    framework.residual_weight=0.05 \\
    framework.max_iters=2 \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
	optimizer.step_size={value} \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    values = [1.0, 0.5, 0.2, 0.1, 0.0]
    prefixs = float_to_scientific(values)
    group_name = "train_vertices"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    framework.residual_weight=0.05 \\
    framework.max_iters=2 \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
    optimizer.step_size=0.7 \\
    framework.vertices_weight={value}
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    values = [1, 2, 3]
    prefixs = float_to_scientific(values)
    group_name = "train_iterations"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    framework.residual_weight=0.05 \\
    framework.max_iters={value} \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
    optimizer.step_size=0.7 \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    values = [0, 1, 2, 3, 4]
    prefixs = float_to_scientific(values)
    group_name = "train_frame_jumps"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    framework.residual_weight=0.05 \\
    framework.max_iters=2 \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
    optimizer.step_size=0.7 \\
    data.train_dataset.jump_size={value} \\
    data.train_dataset.mode=fix \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    values = ["neural", "neural_face2face"]
    prefixs = float_to_scientific(values)
    group_name = "train_residuals"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    residuals={value} \\
    framework.residual_weight=0.05 \\
    framework.max_iters=2 \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
    optimizer.step_size=0.7 \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    values = ["optimization"]
    prefixs = float_to_scientific(values)
    group_name = "train"
    template_generator = """
    python scripts/train.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    data=kinect \\
    residuals=neural_face2face \\
    framework.residual_weight=0.05 \\
    framework.max_iters=2 \\
    framework.max_optims=1 \\
    trainer.max_epochs=50 \\
    optimizer.step_size=0.3 \\
    data.scale=4 \\
    weighting.size=512 \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    with open(f"Makefile.{SUFFIX}", "w") as f:
        f.write(build_makefile(groups))


if __name__ == "__main__":
    main()
