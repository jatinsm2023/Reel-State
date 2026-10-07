# Deploying Reel State on Render

You end up with a public HTTPS link to the site (Try it, How it works, Report) backed by a Render Postgres database with pgvector. About 20 minutes the first time.

**What Render hosts:** one web service (the Python app) and one Postgres database. The film catalogue (13,174 films with their vectors) is loaded into the database once, from your laptop.

## Before you start

* A GitHub account and a Render account (render.com).
* `git` on your computer.
* The local database running (`scripts/db.sh start`) so the catalogue can be exported.

## 1. Put the project on GitHub

```bash
cd /Users/jatin/Documents/MovieRec
git init -b main
git add -A
git status          # check: no .env, no data/, no .venv, no catalog.sql.gz in the list
git commit -m "Reel State"
```

Create an empty repository on github.com (private is fine), then:

```bash
git remote add origin https://github.com/YOUR-NAME/reel-state.git
git push -u origin main
```

`.env` (it holds your TMDB key), `data/` (1.4 GB of MovieLens) and `.venv` are in `.gitignore`. The repository is about 2 MB.

## 2. Create the services from the Blueprint

1. In Render: **New → Blueprint**, connect GitHub, choose the repository.
2. Render reads `render.yaml` and shows two resources: **reelstate-db** (Postgres, free) and **reel-state** (web service, free). Click **Apply**.
3. Wait for the first deploy (a few minutes). On start, the app creates all tables and the pgvector extension by itself, and loads the 20 quiz questions.

The site will load but the film catalogue is empty until step 3.

## 3. Load the film catalogue (once)

In Render open **reelstate-db → Connect → External** and copy the **External Database URL**.

```bash
scripts/export_catalog.sh
```
```bash
scripts/import_catalog.sh "PASTE-THE-EXTERNAL-DATABASE-URL-HERE"
```

Export takes seconds (29 MB), import about a minute. It ends by printing `13174 | 13174`. The import refuses to run if films are already there, so it cannot duplicate anything.

## 4. Check it

Open `https://YOUR-SERVICE.onrender.com/api/status`. You want:

```json
{"pgvector":"0.8.x","iterative_scan_supported":true,"movies":13174,"movies_with_vectors":13174,"ready":true}
```

Then open the site root and try the demo.

## 5. Close the database to the internet

`render.yaml` opens the database to all addresses only so step 3 can work. Close it:

1. In `render.yaml`, replace the `ipAllowList:` block under the database with `ipAllowList: []`.
2. Commit and push (`git commit -am "close database"`, `git push`).
3. In Render, open **reelstate-db → Networking / Access Control** and confirm no `0.0.0.0/0` rule remains.

The web service still reaches the database over Render's private network.

## Things to know

| Item | What happens |
|---|---|
| Free web service | Sleeps after 15 minutes without visitors; the next visit takes about a minute to wake. |
| Free database | 1 GB (we use well under 200 MB). **It expires 30 days after creation**, with 14 days to upgrade before data is deleted. Afterwards, upgrade it, or create a new one and repeat steps 2–3. Profiles and check-ins would be lost. |
| Updating the site | `git push`. Render redeploys automatically. |
| Visitors' data | The Try it page stores each visitor's mood readings and check-ins in your database. Do not share the link widely, or run the pilot on it, until you have the consent process in `pilot/PILOT.md` sorted. |
| pgvector version | Render does not state which version it provides. If `/api/status` shows `iterative_scan_supported: false`, the site still works, but searches with strict filters may return fewer films. |

## If something is wrong

* **Deploy fails at build:** check the build log; the build installs `requirements-web.txt` (no PyTorch needed).
* **`/api/status` errors or `pgvector` is null:** open the service's **Logs**; look for `startup database setup failed`.
* **`ready` is false:** the catalogue is not loaded (step 3), or only partly. Check the printed counts.
* **Site opens but recommendations error:** same as above; the catalogue must be loaded.
* **Cannot connect from your laptop in step 3:** use the **External** URL, not Internal, and check the database still allows your IP (step 5 closes it).
