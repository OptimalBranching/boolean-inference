# One-command setup of the CnC baseline tools.
#
#   make            build/fetch every tool into bin/ + clone proofix/
#   make check      verify each tool responds
#   make clean      remove temporary build trees (keeps bin/ + proofix/)
#   make distclean  remove bin/ and proofix/ too (full reset)
#
# Pinned tools are built into revision-qualified paths and the convenience
# names are refreshed from those exact binaries on every make invocation. A
# revision change therefore creates a new prerequisite instead of silently
# accepting a stale bin/kissat or bin/march_cu. Source trees remain temporary.

BIN     := $(CURDIR)/bin
BUILD   := $(CURDIR)/.build

CADICAL_REPO := https://github.com/arminbiere/cadical.git
KISSAT_REPO  := https://github.com/arminbiere/kissat.git
CNC_REPO     := https://github.com/marijnheule/CnC.git
PROOFIX_REPO := https://github.com/zaxioms0/proofix.git

# Frozen source revisions for the issue #51 hard-regime table. Update these
# only by revising the experiment contract before any held-out run.
KISSAT_REV := 8af8e56f174b778aef3aa45af9f739b2a5f492c2
CNC_REV    := 705b60c6491ef2b61988b3ce6ac674be1b90571d
KISSAT_PINNED := $(BIN)/kissat-$(KISSAT_REV)
CNC_PINNED    := $(BIN)/march_cu-$(CNC_REV)

.PHONY: all check clean distclean FORCE bin/kissat bin/march_cu

all: $(BIN)/cadical $(BIN)/kissat $(BIN)/march_cu proofix
	@rm -rf $(BUILD)
	@echo
	@echo "cnc-tools ready. Put the binaries on PATH:"
	@echo "    export PATH=$(BIN):\$$PATH"

# Relative aliases make explicit targets behave like the default build even
# though the canonical recipes use absolute paths.
bin/kissat: $(BIN)/kissat

bin/march_cu: $(BIN)/march_cu

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
$(BIN)/kissat: $(KISSAT_PINNED) FORCE | $(BIN)
	@cmp -s $< $@ || cp $< $@

$(KISSAT_PINNED): | $(BIN)
	rm -rf $(BUILD)/kissat
	git clone $(KISSAT_REPO) $(BUILD)/kissat
	cd $(BUILD)/kissat && git checkout --detach $(KISSAT_REV)
	cd $(BUILD)/kissat && ./configure && $(MAKE)
	cp $(BUILD)/kissat/build/kissat $@
	rm -rf $(BUILD)/kissat

# march_cu — the lookahead cuber baseline (bundled in marijnheule/CnC).
$(BIN)/march_cu: $(CNC_PINNED) FORCE | $(BIN)
	@cmp -s $< $@ || cp $< $@

$(CNC_PINNED): | $(BIN)
	rm -rf $(BUILD)/CnC
	git clone $(CNC_REPO) $(BUILD)/CnC
	cd $(BUILD)/CnC && git checkout --detach $(CNC_REV)
	cd $(BUILD)/CnC/march_cu && $(MAKE)
	cp $(BUILD)/CnC/march_cu/march_cu $@
	rm -rf $(BUILD)/CnC

# Proofix — the proof-prefix cuber (pure Python, needs cadical on PATH).
proofix:
	git clone --depth 1 $(PROOFIX_REPO) proofix

check:
	@printf "cadical  "; $(BIN)/cadical --version
	@printf "kissat   "; $(KISSAT_PINNED) --version
	@printf "march_cu "; $(CNC_PINNED) 2>&1 | head -1
	@test -f proofix/proofix.py && echo "proofix  present (proofix/proofix.py)" || echo "proofix  MISSING"

clean:
	rm -rf $(BUILD)

distclean: clean
	rm -rf $(BIN) proofix
