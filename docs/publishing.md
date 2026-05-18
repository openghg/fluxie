# FLUXIE - Publishing a New Version

This project is published to PyPI via GitHub Actions Trusted Publishing.

The release workflow is defined in `.github/workflows/python-app.yml`. It publishes automatically when a Git tag is pushed and passes validation. The workflow publishes to PyPI using OpenID Connect (OIDC); no API token is required. See the `release` job in `python-app.yml`.

---

## Release Trigger Rules

The publish job runs only when **all** of these conditions are true:

*   **Event:** The workflow is triggered by a tag push.
*   **Prefix:** The tag starts with `v`.
*   **Format:** The tag contains at least one dot (`.`).
*   **Exclusion:** The tag does *not* contain a hyphen (`-`).

### Examples
*   ✅ **Valid:** `v1.2.0`
*   ❌ **Not published by this workflow:** `v1.2.0-rc1`

---

## Steps for Developers to Publish a New Version

### 1. Prepare and Verify
1. Prepare the release commit on your devel branch.
2. Run local checks before tagging:
    ```bash
    black --check ./tests ./fluxie
    pytest
    ```
3. Ensure all release changes are committed and pushed to the remote repository.

### 2. Tag and Push
1. Create an annotated tag using semantic versioning:
    ```bash
    git tag -a vX.Y.Z -m "Release vX.Y.Z"
    ```
2. Push the tag to GitHub:
    ```bash
    git push origin vX.Y.Z
    ```

### 3. Monitor and Verify
1. Watch the workflow run in GitHub Actions:
    *   The `Build/test` job must pass successfully.
    *   The `release_pypi` job builds the package using `uv build` and publishes it via `uv publish --trusted-publishing always`.
2. Verify the release live on PyPI at [pypi.org/project/fluxie](https://pypi.org/project/fluxie/).

---

## Troubleshooting: If the Release Fails

If an error occurs during the automation pipeline, apply the following steps:

1. Fix the underlying issue directly on the release commit.
2. If you need to retag the release, clear the failed tag entirely:
    ```bash
    # Delete the local tag
    git tag -d vX.Y.Z

    # Delete the remote tag from GitHub
    git push origin :refs/tags/vX.Y.Z
    ```
3. Re-create and push the tag again following the steps in the deployment section.
4. Re-check the GitHub Actions pipeline and PyPI package status.

---

## Versioning Notes

*   **Dynamic Resolution:** The project version is dynamically derived from Git tags through `uv-dynamic-versioning`.
*   **Tag Patterns:** Use tags formatted exactly like `v1.2.3` unless you explicitly configure a different pattern.
*   **Custom Configurations:** If you want tags without a `v` prefix, configure `pattern = "default-unprefixed"` inside your `pyproject.toml` file.
