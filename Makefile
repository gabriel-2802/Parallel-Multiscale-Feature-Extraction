USER        ?= your_user

MAIN_DIR    ?= main_dir
EXEC        ?= a.out

REMOTE_BASE = ~/vtune_automation
REMOTE_DIR  = $(REMOTE_BASE)
REMOTE_MAIN = $(REMOTE_BASE)/$(MAIN_DIR)

RESULT_DIR  = results/$(EXEC)_results
FEP         = fep.grid.pub.ro

run:
	@mkdir -p $(RESULT_DIR)

	@echo ">>> Cleaning remote directory..."
	ssh $(USER)@$(FEP) "rm -rf $(REMOTE_DIR); mkdir -p $(REMOTE_DIR)"

	@echo ">>> Uploading MAIN_DIR, helpers, images..."
	scp -r $(MAIN_DIR) $(USER)@$(FEP):$(REMOTE_DIR)/
	scp -r helpers $(USER)@$(FEP):$(REMOTE_DIR)/
	scp -r images $(USER)@$(FEP):$(REMOTE_DIR)/
	scp remote_run.sh $(USER)@$(FEP):$(REMOTE_DIR)/

	@echo ">>> Running remote build + VTune..."
	ssh $(USER)@$(FEP) "cd $(REMOTE_DIR) && bash remote_run.sh $(MAIN_DIR) $(EXEC)"

	@echo ">>> Downloading VTune results..."
	scp -r $(USER)@$(FEP):$(REMOTE_MAIN)/r* $(RESULT_DIR)/

	@echo ">>> Deleting remote project..."
	ssh $(USER)@$(FEP) "rm -rf $(REMOTE_DIR)"

	@echo "All results saved to: $(RESULT_DIR)/"
