# Anonymous GitHub Publishing Guide

The public repository and GitHub Pages site must not be published from an account that identifies the authors during double-blind review.

## 1. Create an anonymous surface

Use a new GitHub account or organization that has no identifying name, avatar, biography, social links, public email, prior repositories or membership connection to the authors. Use a neutral repository name such as `isimoe-anonymous`.

Do not fork an author-owned repository: the fork relationship can expose provenance. Create a new empty repository instead.

## 2. Configure anonymous commit metadata

Set repository-local metadata before the first commit. Use a GitHub-provided no-reply address belonging to the anonymous account.

```shell
git config user.name "Anonymous Authors"
git config user.email "ANONYMOUS_ACCOUNT_ID+ANONYMOUS_LOGIN@users.noreply.github.com"
git config --local --list
```

Replace the placeholder address with the no-reply address shown by the anonymous account. Do not reuse an institutional or personal email.

## 3. Review tracked files

```shell
git status --short
git grep -n -I -E "author|affiliation|university|institute|@|/home/|Users\\\\"
git ls-files
```

Inspect image, PDF and archive metadata separately. Ensure that raw data, model checkpoints, logs, editor folders and cached bytecode are not tracked.

## 4. Create the first anonymous commit

```shell
git add .
git commit -m "Release anonymous ISI-MoE review artifact"
git branch -M main
git remote add origin https://github.com/ANONYMOUS_LOGIN/isimoe-anonymous.git
git push -u origin main
```

## 5. Enable GitHub Pages

In the anonymous repository, open **Settings → Pages**. Under **Build and deployment**, choose **Deploy from a branch**, select `main` and `/docs`, then save.

The project URL will normally be:

```text
https://ANONYMOUS_LOGIN.github.io/isimoe-anonymous/
```

Open the URL in a signed-out private browser window and test the method figure and artifact download.

## 6. Final double-blind check

- Repository owner, profile, commit authors and commit emails are anonymous.
- Repository history did not originate from a personal repository or fork.
- README, Pages, images, archive and logs contain no identity-bearing material.
- The submission PDF refers to the anonymous URL only.
- Code and supplementary material are available during review, not promised only after acceptance.
