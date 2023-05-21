import subprocess
import requests
import configparser


def get_current_version():
    """ Returns the current version of the application. """
    config = configparser.ConfigParser()
    config.read('config.ini')
    current_version = config.get('Application', 'Version')
    return current_version


def check_for_updates():
    """Checks for updates to the application."""
    owner = "samuel-wechsler"
    repo = "TCID50-parser"

    # Get all releases for the repository
    url = f"https://api.github.com/repos/{owner}/{repo}/releases"
    headers = {
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        releases = response.json()
        latest_release = max(releases, key=lambda r: r["created_at"])
        latest_version = latest_release["tag_name"]

        # Replace with your application's current version
        current_version = get_current_version()

        return not (latest_version == current_version)

    else:
        # Unable to fetch release information
        raise Exception("Failed to fetch release information.")


def update_application():
    """Updates the application by pulling the latest changes from the Git repository."""
    try:
        # Pull the latest changes from the Git repository
        subprocess.run(["git", "pull"])
    except Exception as e:
        raise Exception(f"Failed to update the application: {str(e)}")
