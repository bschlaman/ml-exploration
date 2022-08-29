import logging
from colorama import Fore
from law_large_numbers import lln

logging.basicConfig(
    # format="%(asctime)s [%(levelname)-8s] (%(name)s) %(message)s",
    format="[%(levelname)-8s] (%(name)s) %(message)s",
    level=logging.DEBUG,
)
log = logging.getLogger(__name__)


def main():
    log.info(f"{Fore.BLUE}Welcome to the ml module{Fore.RESET}")
    log.info(f"starting expected value convergence test...")
    lln.lln_convergence()


if __name__ == "__main__":
    main()
