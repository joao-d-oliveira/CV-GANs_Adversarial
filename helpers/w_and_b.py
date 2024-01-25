import wandb


def record_losses_plot(loss, y_label, title):
    x_values = list(range(len(loss)))
    data = [[x, y] for (x, y) in zip(x_values, loss)]

    table = wandb.Table(data=data, columns=["x", y_label])
    wandb.log(
        {
            f"{y_label}_plot_id": wandb.plot.line(
                table, "x", y_label, title=title
            )
        }
    )
