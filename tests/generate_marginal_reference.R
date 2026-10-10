# Independent reference: synthetic data only; requires emmeans, lmerTest, jsonlite.
# Run from the lme-rs-agent repository. R is validation tooling, not a runtime dependency.
.libPaths(c(file.path(Sys.getenv("TEMP"), "lme-mcp-Rlibs"), .libPaths()))
suppressPackageStartupMessages(library(emmeans))
suppressPackageStartupMessages(library(lmerTest))
d <- expand.grid(subject = 1:16, visit = 0:8)
d$A <- factor(c("a", "b", "c")[(d$visit + d$subject) %% 3 + 1])
d$B <- factor(c("low", "high")[(d$visit %/% 3 + d$subject) %% 2 + 1], levels=c("high", "low"))
d$x <- (d$visit - 4) / 4 + sin(d$subject) / 3
d$y <- 4 + c(a=0,b=0.5,c=-0.4)[d$A] + ifelse(d$B=="high", 0.6, 0) +
  ifelse(d$A=="b" & d$B=="high", 0.3, 0) + 0.8*d$x +
  1.2*sin(d$subject*0.7) + 0.6*cos(d$visit*1.3 + d$subject*0.9)
d <- d[!(d$A=="c" & d$B=="low" & d$subject %% 3==0), ]
d$subject <- factor(d$subject)
write.csv(d, "tests/data/marginal.csv", row.names=FALSE, quote=FALSE)
models <- list(lm=lm(y~A*B+x,d), lmm=lmer(y~A*B+x+(1|subject),d,REML=TRUE))
cases <- list()
for (kind in names(models)) {
  for (weight in c("equal", "proportional")) {
    for (conditioned in c(FALSE, TRUE)) {
      by <- if(conditioned) "B" else character()
      e <- emmeans(models[[kind]], specs="A", by=if(conditioned) by else NULL, at=list(x=0.25), weights=weight,
                   lmer.df="satterthwaite")
      means <- as.data.frame(summary(e, level=0.95))
      # Engine reports later cell minus earlier; reverse the R pairwise direction.
      pairs <- as.data.frame(summary(pairs(e, reverse=TRUE, adjust="holm")))
      cases[[length(cases)+1]] <- list(kind=kind, weights=weight, by=by,
                                     means=means, pairs=pairs)
    }
  }
}
jsonlite::write_json(list(provenance=list(R=R.version.string,
  emmeans=as.character(packageVersion("emmeans")), lmerTest=as.character(packageVersion("lmerTest")),
  formula="y ~ A * B + x (+ (1 | subject) for LMM)", seed="none; deterministic synthetic observations"),
  cases=cases), "tests/data/marginal_reference.json", pretty=TRUE, auto_unbox=TRUE, digits=16)
