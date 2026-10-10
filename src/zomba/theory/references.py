"""Sources that the bounds in ``zomba.theory.bounds`` are taken from."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Reference:
    key: str
    authors: str
    title: str
    venue: str
    year: int
    url: str = ""

    def __str__(self):
        return f"{self.authors} ({self.year}). {self.title}. {self.venue}."


def _ref(key, authors, title, venue, year, url=""):
    return Reference(key, authors, title, venue, year, url)


REFERENCES = {r.key: r for r in [
    _ref("ACF2002", "Auer, Cesa-Bianchi & Fischer", "Finite-time analysis of the multiarmed bandit problem",
         "Machine Learning 47:235-256", 2002, "https://doi.org/10.1023/A:1013689704352"),
    _ref("LS2020", "Lattimore & Szepesvari", "Bandit Algorithms", "Cambridge University Press", 2020,
         "https://tor-lattimore.com/downloads/book/book.pdf"),
    _ref("LR1985", "Lai & Robbins", "Asymptotically efficient adaptive allocation rules",
         "Advances in Applied Mathematics 6(1):4-22", 1985),
    _ref("GC2011", "Garivier & Cappe", "The KL-UCB algorithm for bounded stochastic bandits and beyond",
         "COLT 2011, PMLR 19:359-376", 2011, "https://proceedings.mlr.press/v19/garivier11a.html"),
    _ref("AG2017", "Agrawal & Goyal", "Near-optimal regret bounds for Thompson sampling",
         "Journal of the ACM 64(5)", 2017, "https://doi.org/10.1145/3088510"),
    _ref("AYPS2011", "Abbasi-Yadkori, Pal & Szepesvari", "Improved algorithms for linear stochastic bandits",
         "NeurIPS 2011", 2011,
         "https://proceedings.neurips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html"),
    _ref("AG2013", "Agrawal & Goyal", "Thompson sampling for contextual bandits with linear payoffs",
         "ICML 2013, PMLR 28(3):127-135", 2013, "https://proceedings.mlr.press/v28/agrawal13.html"),
    _ref("AG2013v4", "Agrawal & Goyal", "Thompson sampling for contextual bandits with linear payoffs (v4)",
         "arXiv:1209.3352v4", 2014, "https://arxiv.org/abs/1209.3352"),
]}
