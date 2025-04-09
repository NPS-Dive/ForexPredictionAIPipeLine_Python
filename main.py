# Entry Point (Presentation Layer)
# Single Responsibility: Launch the CLI

from presentation.cli import ForexCLI

if __name__ == "__main__":
    cli = ForexCLI()
    cli.run()