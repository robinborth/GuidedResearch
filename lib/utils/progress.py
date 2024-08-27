def reset_progress(progress, total: int):
    progress.n = 0
    progress.last_print_n = 0
    progress.total = total
    progress.refresh()


def close_progress(progresses):
    for p in progresses:
        p.close()
