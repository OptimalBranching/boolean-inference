# One-command setup of the CnC baseline tools.
#
#   make            build/fetch every tool into bin/ + clone proofix/
#   make check      verify each tool responds
#   make clean      remove temporary build trees (keeps bin/ + proofix/)
#   make distclean  remove bin/ and proofix/ too (full reset)
#
# Idempotent: a tool already present in bin/ is left untouched (delete it, or
# run `make distclean`, to force a rebuild). Each tool is built from upstream
# source into a scratch dir, the binary copied into bin/, and the source tree
# removed — so only the ~2 MB of binaries (self-contained: libc++/libSystem
# only) and the small proofix/ Python clone remain. The whole directory is
# git-ignored via .git/info/exclude.

BIN     := $(CURDIR)/bin
BUILD   := $(CURDIR)/.build

CADICAL_REPO := https://github.com/arminbiere/cadical.git
KISSAT_REPO  := https://github.com/arminbiere/kissat.git
CNC_REPO     := https://github.com/marijnheule/CnC.git
PROOFIX_REPO := https://github.com/zaxioms0/proofix.git

.PHONY: all check clean distclean

all: $(BIN)/cadical $(BIN)/kissat $(BIN)/march_cu proofix
	@rm -rf $(BUILD)
	@echo
	@echo "cnc-tools ready. Put the binaries on PATH:"
	@echo "    export PATH=$(BIN):\$$PATH"

$(BIN):
	mkdir -p $(BIN)

# CaDiCaL — conquer solver + DRAT emitter that Proofix drives.
$(BIN)/cadical: | $(BIN)
	rm -rf $(BUILD)/cadical
	git clone --depth 1 $(CADICAL_REPO) $(BUILD)/cadical
	cd $(BUILD)/cadical && ./configure && $(MAKE)
	cp $(BUILD)/cadical/build/cadical $@
	rm -rf $(BUILD)/cadical

# kissat — the fixed conquer solver for every cuber (fairness).
$(BIN)/kissat: | $(BIN)
	rm -rf $(BUILD)/kissat
	git clone --depth 1 $(KISSAT_REPO) $(BUILD)/kissat
	cd $(BUILD)/kissat && ./configure && $(MAKE)
	cp $(BUILD)/kissat/build/kissat $@
	rm -rf $(BUILD)/kissat

# march_cu — the lookahead cuber baseline (bundled in marijnheule/CnC).
$(BIN)/march_cu: | $(BIN)
	rm -rf $(BUILD)/CnC
	git clone --depth 1 $(CNC_REPO) $(BUILD)/CnC
	cd $(BUILD)/CnC/march_cu && $(MAKE)
	cp $(BUILD)/CnC/march_cu/march_cu $@
	rm -rf $(BUILD)/CnC

# Proofix — the proof-prefix cuber (pure Python, needs cadical on PATH).
proofix:
	git clone --depth 1 $(PROOFIX_REPO) proofix

check:
	@printf "cadical  "; $(BIN)/cadical --version
	@printf "kissat   "; $(BIN)/kissat --version
	@printf "march_cu "; $(BIN)/march_cu 2>&1 | head -1
	@test -f proofix/proofix.py && echo "proofix  present (proofix/proofix.py)" || echo "proofix  MISSING"

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(BIN) proofix
