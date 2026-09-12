#!/usr/bin/env Rscript

# Publication figures for held-out GP validation and fold-wise ARD screening.
# Uses base R graphics so the reporting package has no additional dependencies.

args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
asset_dir <- dirname(script_path)
results_dir <- asset_dir
figures_dir <- asset_dir

pred <- read.csv(file.path(results_dir, "surrogate_holdout_predictions.csv"), check.names = FALSE)
summary <- read.csv(file.path(results_dir, "surrogate_validation_summary.csv"), check.names = FALSE)
ard <- read.csv(file.path(results_dir, "ard_fold_screening.csv"), check.names = FALSE)

padded_limits <- function(values) {
  limits <- range(values, finite = TRUE)
  pad <- 0.055 * diff(limits)
  limits + c(-pad, pad)
}

scientific_parts <- function(value) {
  exponent <- floor(log10(abs(value)))
  list(mantissa = value / (10^exponent), exponent = exponent)
}

parity_panel <- function(year, limits, panel_label, show_y = TRUE,
                         bottom_caption = FALSE) {
  d <- pred[pred$year == year, ]
  s <- summary[summary$year == year, ][1, ]

  plot(
    d$uced_loss, d$gp_predicted_loss,
    xlim = limits, ylim = limits, asp = 1,
    type = "n",
    xlab = expression(paste("Original-UCED mismatch, ", L(theta))),
    ylab = if (show_y) expression(paste("GP-predicted mismatch, ", hat(L)(theta))) else "",
    main = "", family = "serif", las = 1
  )

  # Relative-error reference region: 0.95 L <= L-hat <= 1.05 L.
  band_x <- seq(limits[1], limits[2], length.out = 240)
  polygon(
    c(band_x, rev(band_x)),
    c(0.95 * band_x, rev(1.05 * band_x)),
    border = NA,
    col = adjustcolor("#4C78A8", alpha.f = 0.08)
  )

  # Major grid only, kept deliberately faint so it does not compete with the band.
  abline(v = axTicks(1), h = axTicks(2), col = adjustcolor("grey65", alpha.f = 0.18), lwd = 0.42)
  abline(a = 0, b = 1, lty = 2, col = "grey35", lwd = 1)

  points(
    d$uced_loss, d$gp_predicted_loss,
    pch = 21, bg = "#1768AC", col = "white", cex = 0.83, lwd = 0.48
  )

  # Compact journal-style annotation with true mathematical scientific notation.
  rmse_parts <- scientific_parts(s$test_rmse)
  stat_line_1 <- bquote(R[test]^2 == .(formatC(s$test_r2, format = "f", digits = 3)))
  stat_line_2 <- parse(
    text = sprintf("RMSE == '%0.2f' %%*%% 10^{%d}", rmse_parts$mantissa, rmse_parts$exponent)
  )[[1]]
  usr <- par("usr")
  stat_x <- usr[1] + 0.052 * diff(usr[1:2])
  stat_y <- usr[4] - 0.065 * diff(usr[3:4])
  text(stat_x, stat_y, stat_line_1, adj = c(0, 1), cex = 0.73, family = "serif")
  text(stat_x, stat_y - 0.075 * diff(usr[3:4]), stat_line_2,
       adj = c(0, 1), cex = 0.73, family = "serif")

  if (bottom_caption) {
    # Keep the year and panel label together below the x-axis title.  Moving
    # this information out of the plotting area also removes the unused top
    # title margin in the compact two-column version.
    panel_caption <- bquote(bold(.(panel_label)) ~ .(as.character(year)))
    mtext(panel_caption, side = 1, line = 2.80, cex = 0.80,
          font = 1, family = "serif")
  } else {
    title(main = as.character(year), line = 0.20, cex.main = 0.86,
          font.main = 1, family = "serif")
    mtext(panel_label, side = 3, adj = 0.01, line = 0.20, cex = 0.80,
          font = 2, family = "serif")
  }
  box(lwd = 0.7)
}

pdf(file.path(figures_dir, "fig_gp_holdout_validation.pdf"), width = 7.05, height = 3.35,
    family = "Times", useDingbats = FALSE)
par(mfrow = c(1, 2), mar = c(4.15, 3.70, 0.45, 0.30), oma = c(0, 0, 0, 0),
    mgp = c(1.75, 0.35, 0), tcl = -0.22, cex = 0.90, cex.axis = 0.86, cex.lab = 0.92)
for (i in seq_along(c(2016, 2021))) {
  year <- c(2016, 2021)[i]
  d <- pred[pred$year == year, ]
  parity_panel(year, padded_limits(c(d$uced_loss, d$gp_predicted_loss)),
               c("(a)", "(b)")[i], i == 1, bottom_caption = TRUE)
}
dev.off()

png(file.path(figures_dir, "fig_gp_holdout_validation.png"), width = 7.05, height = 3.35,
    units = "in", res = 600, type = "cairo", family = "serif")
par(mfrow = c(1, 2), mar = c(4.15, 3.70, 0.45, 0.30), oma = c(0, 0, 0, 0),
    mgp = c(1.75, 0.35, 0), tcl = -0.22, cex = 0.90, cex.axis = 0.86, cex.lab = 0.92)
for (i in seq_along(c(2016, 2021))) {
  year <- c(2016, 2021)[i]
  d <- pred[pred$year == year, ]
  parity_panel(year, padded_limits(c(d$uced_loss, d$gp_predicted_loss)),
               c("(a)", "(b)")[i], i == 1, bottom_caption = TRUE)
}
dev.off()

pdf(file.path(figures_dir, "fig_gp_holdout_validation_stacked.pdf"), width = 3.5, height = 6.45,
    family = "Times", useDingbats = FALSE)
par(mfrow = c(2, 1), mar = c(3.75, 3.90, 1.35, 1.40), oma = c(0, 0, 0, 0),
    mgp = c(2.25, 0.60, 0), tcl = -0.22, cex = 0.88, cex.axis = 0.84, cex.lab = 0.90)
for (i in seq_along(c(2016, 2021))) {
  year <- c(2016, 2021)[i]
  d <- pred[pred$year == year, ]
  parity_panel(year, padded_limits(c(d$uced_loss, d$gp_predicted_loss)), c("(a)", "(b)")[i], TRUE)
}
dev.off()

png(file.path(figures_dir, "fig_gp_holdout_validation_stacked.png"), width = 3.5, height = 6.45,
    units = "in", res = 600, type = "cairo", family = "serif")
par(mfrow = c(2, 1), mar = c(3.75, 3.90, 1.35, 1.40), oma = c(0, 0, 0, 0),
    mgp = c(2.25, 0.60, 0), tcl = -0.22, cex = 0.88, cex.axis = 0.84, cex.lab = 0.90)
for (i in seq_along(c(2016, 2021))) {
  year <- c(2016, 2021)[i]
  d <- pred[pred$year == year, ]
  parity_panel(year, padded_limits(c(d$uced_loss, d$gp_predicted_loss)), c("(a)", "(b)")[i], TRUE)
}
dev.off()

short_names <- c(
  "Coal_Retrofitted_Min_Power_660-1000" = "Retrofit min., 660--1000 MW",
  "Coal_Retrofitted_Time_660-1000" = "Retrofit time, 660--1000 MW",
  "Coal_Retrofitted_Min_Power_300-660" = "Retrofit min., 300--660 MW",
  "Coal_Retrofitted_Time_300-660" = "Retrofit time, 300--660 MW",
  "Coal_Retrofitted_Min_Power_0-300" = "Retrofit min., 0--300 MW",
  "Coal_Retrofitted_Time_0-300" = "Retrofit time, 0--300 MW",
  "CHP_NonRetrofitted_Min_Power_660-1000" = "CHP min., 660--1000 MW",
  "CHP_NonRetrofitted_Time_660-1000" = "CHP time, 660--1000 MW",
  "CHP_NonRetrofitted_Min_Power_300-660" = "CHP min., 300--660 MW",
  "CHP_NonRetrofitted_Time_300-660" = "CHP time, 300--660 MW",
  "CHP_NonRetrofitted_Min_Power_0-300" = "CHP min., 0--300 MW",
  "CHP_NonRetrofitted_Time_0-300" = "CHP time, 0--300 MW",
  "MLT_Band" = "Physical MLT band"
)

parameter_order <- unique(ard$parameter[ard$year == 2016])
y_positions <- rev(seq_along(parameter_order))

ard_panel <- function(year, panel_label, show_y = TRUE) {
  plot(NA, xlim = c(0.65, 7.65), ylim = c(0.45, length(parameter_order) + 0.55),
       xaxs = "i", yaxs = "i", xaxt = "n", yaxt = "n", xlab = "Fold", ylab = "",
       main = as.character(year), bty = "l", family = "serif")
  axis(1, at = 1:5, labels = paste0("F", 1:5), tck = -0.02, gap.axis = -1, cex.axis = 0.82)
  if (show_y) axis(2, at = y_positions, labels = short_names[parameter_order], las = 1, tick = FALSE, cex.axis = 0.72)
  abline(v = 1:5, col = "grey91", lwd = 0.5)

  for (j in seq_along(parameter_order)) {
    parameter <- parameter_order[j]
    rows <- ard[ard$year == year & ard$parameter == parameter, ]
    rows <- rows[order(rows$fold), ]
    colors <- ifelse(rows$hit_upper_bound, "#C44741", "#2A7F62")
    points(rows$fold, rep(y_positions[j], nrow(rows)), pch = 21, bg = colors, col = "white", cex = 1.05, lwd = 0.55)
    status <- if (rows$retained[1]) "retain" else "exclude"
    text(5.55, y_positions[j], sprintf("%d/5  %s", rows$total_hits[1], status), adj = c(0, 0.5), cex = 0.66)
  }
  mtext(panel_label, side = 3, adj = -0.06, line = 0.55, font = 2, family = "serif")
}

draw_ard <- function(device, filename) {
  device(filename)
  par(mfrow = c(1, 2), mar = c(3.8, 10.4, 2.05, 0.4), oma = c(0, 0, 1.2, 0),
      mgp = c(2.2, 0.65, 0), tcl = -0.22, cex = 0.88)
  ard_panel(2016, "(a)", TRUE)
  par(mar = c(3.8, 0.7, 2.05, 0.4))
  ard_panel(2021, "(b)", FALSE)
  legend("top", inset = c(0, -0.16), xpd = NA, horiz = TRUE, bty = "n", cex = 0.72,
         legend = c("At upper bound", "Below upper bound"), pch = 21,
         pt.bg = c("#C44741", "#2A7F62"), col = "white", pt.cex = 1.05)
  dev.off()
}

draw_ard(
  function(filename) pdf(filename, width = 7.2, height = 5.05, family = "Times", useDingbats = FALSE),
  file.path(figures_dir, "supp_ard_fold_stability.pdf")
)
draw_ard(
  function(filename) png(filename, width = 7.2, height = 5.05, units = "in", res = 600,
                         type = "cairo", family = "serif"),
  file.path(figures_dir, "supp_ard_fold_stability.png")
)

# Compact IEEE two-column version: two held-out parity panels followed by two
# year-specific summaries of the fold-wise ARD decisions. The detailed 5-fold
# marker plot above remains the supplementary version.
year_colors <- c("2016" = "#1768AC", "2021" = "#D55E00")

compact_names <- c(
  "Coal_Retrofitted_Min_Power_660-1000" = "R min L",
  "Coal_Retrofitted_Time_660-1000" = "R time L",
  "Coal_Retrofitted_Min_Power_300-660" = "R min M",
  "Coal_Retrofitted_Time_300-660" = "R time M",
  "Coal_Retrofitted_Min_Power_0-300" = "R min S",
  "Coal_Retrofitted_Time_0-300" = "R time S",
  "CHP_NonRetrofitted_Min_Power_660-1000" = "CHP min L",
  "CHP_NonRetrofitted_Time_660-1000" = "CHP time L",
  "CHP_NonRetrofitted_Min_Power_300-660" = "CHP min M",
  "CHP_NonRetrofitted_Time_300-660" = "CHP time M",
  "CHP_NonRetrofitted_Min_Power_0-300" = "CHP min S",
  "CHP_NonRetrofitted_Time_0-300" = "CHP time S",
  "MLT_Band" = "MLT band"
)

compact_parity_panel <- function(year, panel_label, show_y = TRUE) {
  d <- pred[pred$year == year, ]
  s <- summary[summary$year == year, ][1, ]
  limits <- padded_limits(c(d$uced_loss, d$gp_predicted_loss))
  color <- year_colors[as.character(year)]

  par(mar = c(2.65, if (show_y) 2.85 else 1.75, 1.30, 0.25),
      mgp = c(1.55, 0.38, 0), tcl = -0.18)
  plot(
    d$uced_loss, d$gp_predicted_loss,
    xlim = limits, ylim = limits, asp = 1, type = "n",
    xlab = "UCED loss", ylab = if (show_y) "GP loss" else "",
    family = "serif", las = 1, cex.axis = 0.63, cex.lab = 0.72
  )
  band_x <- seq(limits[1], limits[2], length.out = 200)
  polygon(c(band_x, rev(band_x)), c(0.95 * band_x, rev(1.05 * band_x)),
          border = NA, col = adjustcolor(color, alpha.f = 0.09))
  abline(a = 0, b = 1, lty = 2, col = "grey38", lwd = 0.85)
  points(d$uced_loss, d$gp_predicted_loss, pch = 21, bg = color,
         col = "white", cex = 0.72, lwd = 0.40)

  rmse_parts <- scientific_parts(s$test_rmse)
  usr <- par("usr")
  text(usr[1] + 0.05 * diff(usr[1:2]), usr[4] - 0.06 * diff(usr[3:4]),
       bquote(R^2 == .(formatC(s$test_r2, format = "f", digits = 3))),
       adj = c(0, 1), cex = 0.72, family = "serif")
  text(usr[1] + 0.05 * diff(usr[1:2]), usr[4] - 0.145 * diff(usr[3:4]),
       parse(text = sprintf("RMSE == '%0.2f' %%*%% 10^{%d}",
                            rmse_parts$mantissa, rmse_parts$exponent))[[1]],
       adj = c(0, 1), cex = 0.72, family = "serif")
  title(main = sprintf("%s  %d", panel_label, year), line = 0.20,
        cex.main = 0.78, font.main = 2, family = "serif")
  box(lwd = 0.65)
}

compact_ard_panel <- function(year, panel_label, show_y = TRUE) {
  d <- ard[ard$year == year, ]
  d <- d[!duplicated(d$parameter), ]
  d <- d[match(parameter_order, d$parameter), ]
  ypos <- y_positions
  color <- year_colors[as.character(year)]

  par(mar = c(2.65, if (show_y) 4.00 else 0.80, 1.30, 0.25),
      mgp = c(1.55, 0.38, 0), tcl = -0.18)
  plot(NA, xlim = c(-0.25, 5.25), ylim = c(0.45, length(parameter_order) + 0.55),
       xaxs = "i", yaxs = "i", xaxt = "n", yaxt = "n",
       xlab = "Upper-bound hits", ylab = "", family = "serif")
  rect(3.5, par("usr")[3], 5.25, par("usr")[4], border = NA,
       col = adjustcolor("grey55", alpha.f = 0.08))
  abline(v = 3.5, lty = 3, col = "grey55", lwd = 0.70)
  abline(h = ypos, col = adjustcolor("grey70", alpha.f = 0.18), lwd = 0.35)
  axis(1, at = 0:5, labels = 0:5, cex.axis = 0.62)
  if (show_y) {
    axis(2, at = ypos, labels = compact_names[parameter_order], las = 1,
         tick = FALSE, cex.axis = 0.62, line = -0.25)
  }
  points(d$total_hits, ypos, pch = 21,
         bg = ifelse(d$retained, color, "white"),
         col = ifelse(d$retained, color, "grey58"),
         cex = 0.78, lwd = 0.75)
  title(main = sprintf("%s  %d ARD", panel_label, year), line = 0.20,
        cex.main = 0.78, font.main = 2, family = "serif")
  box(lwd = 0.65)
}

draw_ieee_four_panel <- function(device, filename) {
  device(filename)
  # Panel (c) needs extra allocation for its y-axis labels, but the remaining
  # plotting rectangles of (c) and (d) should have the same physical width.
  layout(matrix(1:4, nrow = 1), widths = c(1.03, 1.03, 1.27, 1.03))
  par(oma = c(0, 0, 0, 0), family = "serif")
  compact_parity_panel(2016, "(a)", TRUE)
  compact_parity_panel(2021, "(b)", FALSE)
  compact_ard_panel(2016, "(c)", TRUE)
  compact_ard_panel(2021, "(d)", FALSE)
  dev.off()
}

draw_ieee_four_panel(
  function(filename) pdf(filename, width = 7.16, height = 2.65,
                         family = "Times", useDingbats = FALSE),
  file.path(figures_dir, "fig_surrogate_performance_active_parameters_ieee.pdf")
)
draw_ieee_four_panel(
  function(filename) png(filename, width = 7.16, height = 2.65, units = "in",
                         res = 600, type = "cairo", family = "serif"),
  file.path(figures_dir, "fig_surrogate_performance_active_parameters_ieee.png")
)
