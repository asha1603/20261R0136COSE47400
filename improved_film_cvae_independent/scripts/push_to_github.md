# Push this folder to GitHub

Assuming your repo is already cloned:

```bash
cd 20261R0136COSE47400
mkdir -p FiLM
cp -r /path/to/improved_film_cvae_independent FiLM/
git add FiLM/improved_film_cvae_independent
git commit -m "Add independent improved FiLM CVAE implementation"
git push origin main
```

On Windows PowerShell:

```powershell
cd C:\Users\YOUR_NAME\Downloads\20261R0136COSE47400
mkdir FiLM -ErrorAction SilentlyContinue
Copy-Item -Recurse "$HOME\Downloads\improved_film_cvae_independent" ".\FiLM\"
git add FiLM/improved_film_cvae_independent
git commit -m "Add independent improved FiLM CVAE implementation"
git push origin main
```
