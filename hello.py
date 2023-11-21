import click
import time

@click.command()
@click.argument("name")
@click.option(
    "-c",
    "--count",
    default=1,
    help="Number of times to print greeting.",
    show_default=True,  # show default in help
)

@click.option("--pause", default=0, help="Wachttijd na het printen van elk bericht in seconden.")

def hello(name, count, pause):
    for _ in range(count):
        print(f"Hello {name}!")

        if pause:
            time.sleep(pause)
            print(f"Wachten voor {pause} seconden na het begroeten van {name}.")

if __name__ == "__main__":
    hello()
