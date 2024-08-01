def float_to_scientific(values):
    return [f"{v:.1E}".replace(".", "-") for v in values]


def generate_banner(group_names: list[str]):
    b = ""
    b += "##########################################################################\n"
    b += "# make all -f Makefile.abl\n"
    for i, group in enumerate(group_names):
        b += f"# GROUP{i}: make {group} -f Makefile.abl\n"
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

    values = [
        "[0]",
        "[0,20]",
        "[0,20,35]",
        "[0,20,35,55]",
        "[0,20,35,55,65]",
        "[0,20,35,55,65,95]",
        "[0,20,35,55,65,95,110]",
        "[0,20,35,55,65,95,110,115]",
    ]
    prefixs = [1, 2, 3, 4, 5, 6, 7, 8]
    # values = ["v1", "v2"]
    # prefixs = values

    # group_name = "timer_gn_p2p_v1"
    # template_generator = """
    # python scripts/optimize.py \\
    # logger.group={group_name} \\
    # logger.name={task_name} \\
    # logger.tags=[{group_name},{task_name}] \\
    # task_name={task_name} \\
    # model=[flame,dphm_christoph_mouthmove] \\
    # optimizer=gauss_newton \\
    # loss=point2plane \\
    # +loss.jacobian_fn=v1 \\
    # joint_trainer.init_idxs={value} \\
    # joint_trainer.scheduler.milestones=[0] \\
    # joint_trainer.max_iters=50 \\
    # joint_trainer.max_optims=10 \\
    # joint_trainer.scheduler.params=[[global_pose,transl,neck_pose,shape_params,expression_params]] \\
    # sequential_trainer=null \\
    # """
    # groups.append(build_group(template_generator, values, prefixs, group_name))

    group_name = "timer_gn_p2p_v2_no_shape_single_step"
    template_generator = """
    python scripts/optimize.py \\
    logger.group={group_name} \\
    logger.name={task_name} \\
    logger.tags=[{group_name},{task_name}] \\
    task_name={task_name} \\
    model=[flame,dphm_christoph_mouthmove] \\
	optimizer=gauss_newton \\
	loss=point2plane \\
    +loss.jacobian_fn=v2 \\
	joint_trainer.init_idxs={value} \\
    joint_trainer.scheduler.milestones=[0] \\
    joint_trainer.max_iters=50 \\
	joint_trainer.max_optims=1 \\
    joint_trainer.scheduler.params=[[global_pose,transl,neck_pose,expression_params]] \\
	sequential_trainer=null \\
    """
    groups.append(build_group(template_generator, values, prefixs, group_name))

    with open("Makefile.abl", "w") as f:
        f.write(build_makefile(groups))


if __name__ == "__main__":
    main()
