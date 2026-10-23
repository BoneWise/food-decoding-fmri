% Condition-level first-level GLM for the current food task dataset.
% This paper-like variant adds 6-mm FWHM smoothing before first-level GLM,
% includes WM/CSF nuisance regressors, and models valuation trials as
% rating_1/rating_2/rating_3/rating_4 conditions.

clearvars -except subject_id session_id run_id do_estimate pilot_only pilot_max_target_trials overwrite_existing spm_dir max_parallel_subjects smooth_fwhm_mm extra_nuisance_names delete_subject_smooth_after_success input_root output_root
clc;

% -------------------- edit here --------------------
if ~exist('subject_id', 'var'), subject_id = 'ALL'; end
if ~exist('session_id', 'var'), session_id = 'ALL'; end
if ~exist('run_id', 'var'), run_id = 'ALL'; end
if ~exist('do_estimate', 'var'), do_estimate = true; end
if ~exist('pilot_only', 'var'), pilot_only = false; end
if ~exist('pilot_max_target_trials', 'var'), pilot_max_target_trials = []; end
if ~exist('overwrite_existing', 'var'), overwrite_existing = true; end
if ~exist('spm_dir', 'var'), spm_dir = ''; end
if ~exist('max_parallel_subjects', 'var'), max_parallel_subjects = 5; end
if ~exist('smooth_fwhm_mm', 'var'), smooth_fwhm_mm = 6; end
if ~exist('extra_nuisance_names', 'var'), extra_nuisance_names = {'csf', 'white_matter'}; end
if ~exist('delete_subject_smooth_after_success', 'var'), delete_subject_smooth_after_success = true; end
% --------------------------------------------------

if ~exist('input_root', 'var') || isempty(input_root)
    input_root = getenv('FOODPREF_FMRIPREP_ROOT');
end
if isempty(input_root)
    error('Set input_root or FOODPREF_FMRIPREP_ROOT to the fMRIPrep derivatives directory.');
end
events_root = fullfile(input_root, 'events');
if ~exist('output_root', 'var') || isempty(output_root)
    output_root = getenv('FOODPREF_GLM_S6_ROOT');
end
if isempty(output_root)
    error('Set output_root or FOODPREF_GLM_S6_ROOT to the first-level GLM output directory.');
end
models_root = fullfile(output_root, 'models');
cache_root = fullfile(output_root, 'cache');
index_csv = fullfile(output_root, 'contrast_index.csv');

tr = 0.8;
fmri_t = 60;
fmri_t0 = 30;
hpf = 128;
fd_threshold = 0.9;
head_radius_mm = 50;
motion_names = {'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'};

ensure_dir(output_root);
ensure_dir(models_root);
ensure_dir(cache_root);

if pilot_only
    if ~isscalar(pilot_max_target_trials) || pilot_max_target_trials < 1
        error('pilot_max_target_trials must be a positive integer when pilot_only is true.');
    end
end

[resolved_spm_dir, spm_release] = setup_spm25(spm_dir);
fprintf('SPM release: %s\n', spm_release);
fprintf('SPM dir: %s\n', resolved_spm_dir);
fprintf('Smooth FWHM (mm): %.3f\n', smooth_fwhm_mm);
fprintf('Extra nuisance regressors: %s\n', strjoin(cellstr(string(extra_nuisance_names)), ', '));
fprintf('Delete subject smooth cache after successful subject completion: %d\n', delete_subject_smooth_after_success);

all_specs = discover_available_runs(events_root);
if isempty(all_specs)
    error('No event files were discovered under %s', events_root);
end

selected_specs = filter_run_specs(all_specs, subject_id, session_id, run_id);
if isempty(selected_specs)
    error('No runs matched subject=%s session=%s run=%s', char(string(subject_id)), char(string(session_id)), char(string(run_id)));
end

fprintf('Selected runs: %d\n', numel(selected_specs));
fprintf('Pilot mode: %d\n', pilot_only);
if pilot_only
    fprintf('Pilot max target trials per run: %d\n', pilot_max_target_trials);
end

cfg = struct();
cfg.input_root = input_root;
cfg.models_root = models_root;
cfg.cache_root = cache_root;
cfg.tr = tr;
cfg.fmri_t = fmri_t;
cfg.fmri_t0 = fmri_t0;
cfg.hpf = hpf;
cfg.fd_threshold = fd_threshold;
cfg.head_radius_mm = head_radius_mm;
cfg.motion_names = motion_names;
cfg.extra_nuisance_names = cellstr(string(extra_nuisance_names));
cfg.do_estimate = do_estimate;
cfg.pilot_only = pilot_only;
cfg.pilot_max_target_trials = pilot_max_target_trials;
cfg.overwrite_existing = overwrite_existing;
cfg.spm_dir = resolved_spm_dir;
cfg.smooth_fwhm_mm = smooth_fwhm_mm;
cfg.delete_subject_smooth_after_success = delete_subject_smooth_after_success;

selected_subjects = unique({selected_specs.sub}, 'stable');
subject_specs_cells = cell(numel(selected_subjects), 1);
for subj_idx = 1:numel(selected_subjects)
    subject_specs_cells{subj_idx} = selected_specs(strcmp({selected_specs.sub}, selected_subjects{subj_idx}));
end

fprintf('Selected subjects: %d\n', numel(selected_subjects));

results_cells = cell(numel(selected_subjects), 1);
if can_use_parallel_subjects(max_parallel_subjects, numel(selected_subjects))
    pool_size = min(max_parallel_subjects, numel(selected_subjects));
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= pool_size
        if ~isempty(pool)
            delete(pool);
        end
        parpool('local', pool_size);
    end
    fprintf('Parallel subject workers: %d\n', pool_size);
    parfor subj_idx = 1:numel(selected_subjects)
        results_cells{subj_idx} = run_subject_block(subject_specs_cells{subj_idx}, cfg);
    end
else
    fprintf('Parallel subject workers: 1\n');
    for subj_idx = 1:numel(selected_subjects)
        results_cells{subj_idx} = run_subject_block(subject_specs_cells{subj_idx}, cfg);
    end
end

results = concat_result_cells(results_cells);

index_table = struct2table(force_nonempty_results(results), 'AsArray', true);
writetable(index_table, index_csv);

fprintf('\nDone.\n');
fprintf('Contrast index: %s\n', index_csv);
fprintf('Rows written: %d\n', height(index_table));
fprintf('Models estimated successfully: %d\n', nnz(index_table.model_estimated_successfully));
fprintf('1>4 contrasts available: %d\n', nnz(index_table.contrast_rating1_gt_rating4_available));
fprintf('4>1 contrasts available: %d\n', nnz(index_table.contrast_rating4_gt_rating1_available));


function [resolved_spm_dir, spm_release] = setup_spm25(user_spm_dir)
candidate_dirs = {};

if exist('spm', 'file') == 2
    candidate_dirs{end + 1} = fileparts(which('spm')); %#ok<AGROW>
end
if ~isempty(user_spm_dir)
    candidate_dirs{end + 1} = char(string(user_spm_dir)); %#ok<AGROW>
end

home_dir = getenv('HOME');
candidate_dirs{end + 1} = fullfile(home_dir, 'Documents', 'MATLAB', 'spm25'); %#ok<AGROW>
candidate_dirs{end + 1} = fullfile(home_dir, 'MATLAB', 'spm25'); %#ok<AGROW>

candidate_dirs = unique(candidate_dirs, 'stable');
resolved_spm_dir = '';
spm_release = '';

for i = 1:numel(candidate_dirs)
    candidate = candidate_dirs{i};
    if isempty(candidate) || exist(fullfile(candidate, 'spm.m'), 'file') ~= 2
        continue;
    end

    addpath(candidate, '-begin');

    try
        [release_name, ~] = spm('Ver');
        active_spm_dir = fileparts(which('spm'));
        if is_spm25_candidate(active_spm_dir, release_name)
            spm('Defaults', 'fMRI');
            spm_get_defaults('cmdline', true);
            spm_jobman('initcfg');
            resolved_spm_dir = active_spm_dir;
            spm_release = char(string(release_name));
            return;
        end
    catch
    end
end

error(['SPM25 was not found. Put SPM25 on the MATLAB path first, or set spm_dir ' ...
       'to the SPM25 directory before running this script.']);
end


function tf = is_spm25_candidate(spm_dir, release_name)
release_text = upper(char(string(release_name)));
dir_text = lower(char(string(spm_dir)));
tf = contains(release_text, '25') || contains(dir_text, 'spm25');
end


function tf = can_use_parallel_subjects(max_parallel_subjects, n_subjects)
tf = max_parallel_subjects > 1 && n_subjects > 1 && ...
    license('test', 'Distrib_Computing_Toolbox') && exist('parpool', 'file') == 2;
end


function tf = subject_outputs_complete_for_cleanup(subject_rows)
if isempty(subject_rows)
    tf = false;
    return;
end

tf = true;
for i = 1:numel(subject_rows)
    row = subject_rows(i);

    if ~row.model_estimated_successfully || isempty(row.spm_mat)
        tf = false;
        return;
    end

    has_both_extremes = row.n_rating1 > 0 && row.n_rating4 > 0;
    if has_both_extremes && ...
            ~(row.contrast_rating1_gt_rating4_available && row.contrast_rating4_gt_rating1_available)
        tf = false;
        return;
    end
end
end


function delete_subject_smoothed_cache(cache_root, subject_id)
subject_smooth_dir = fullfile(cache_root, 'smoothed', subject_id);
if exist(subject_smooth_dir, 'dir') ~= 7
    fprintf('Smooth cache already absent: %s\n', subject_smooth_dir);
    return;
end

fprintf('Deleting smooth cache for completed subject: %s\n', subject_smooth_dir);
rmdir(subject_smooth_dir, 's');
end


function subject_rows = run_subject_block(subject_specs, cfg)
setup_spm25(cfg.spm_dir);
subject_rows = struct([]);
subject_id = subject_specs(1).sub;

fprintf('\n=== %s (%d runs) ===\n', subject_id, numel(subject_specs));

for i = 1:numel(subject_specs)
    spec = subject_specs(i);
    fprintf('\n--- %s %s %s ---\n', spec.sub, spec.session_id, spec.run);
    run_rows = run_single_run_lsa(spec, cfg);
    subject_rows = append_rows(subject_rows, run_rows);
end

if cfg.delete_subject_smooth_after_success && subject_outputs_complete_for_cleanup(subject_rows)
    delete_subject_smoothed_cache(cfg.cache_root, subject_id);
end
end


function run_rows = run_single_run_lsa(spec, cfg)
run_rows = struct([]);
spm_mat = '';
selected_target_rows = [];
collapsed_target_rows = [];
valuation_summary = make_empty_valuation_summary();
contrast_lookup = default_condition_contrast_specs();

try
    [bold_file, confounds_file, events_file, mask_file] = resolve_input_files( ...
        cfg.input_root, cfg.cache_root, spec.sub, spec.session_id, spec.run);

    fprintf('BOLD: %s\n', bold_file);
    fprintf('Confounds: %s\n', confounds_file);
    fprintf('Events: %s\n', events_file);
    fprintf('Mask: %s\n', mask_file);

    smoothed_bold_file = prepare_smoothed_bold_if_needed( ...
        bold_file, cfg.cache_root, spec.sub, spec.session_id, spec.run, cfg.smooth_fwhm_mm);
    fprintf('Smoothed BOLD: %s\n', smoothed_bold_file);

    events = readtable(events_file, 'FileType', 'text', 'Delimiter', '\t');
    must_have_columns(events, {'onset', 'duration', 'trial_id', 'trial_type', 'rating_value'});
    events.trial_type = lower(string(events.trial_type));

    [target_rows, ~] = select_nonmiss_target_rows(events);
    if isempty(target_rows)
        run_rows = initialize_run_failure_row(spec, valuation_summary, 0, default_condition_contrast_specs(), '', ...
            'No analyzable non-miss valuation trials were found in this run.');
        return;
    end

    [selected_target_rows, collapsed_target_rows, selection_note] = ...
        select_target_rows_for_run(target_rows, cfg.pilot_only, cfg.pilot_max_target_trials);
    valuation_summary = summarize_selected_valuation_rows(selected_target_rows);

    conditions = build_condition_level_conditions( ...
        events, selected_target_rows, collapsed_target_rows);

    regressors = build_nuisance_regressors( ...
        confounds_file, cfg.motion_names, cfg.extra_nuisance_names, cfg.fd_threshold, cfg.head_radius_mm);
    scans = build_scan_list(smoothed_bold_file);

    model_dir = fullfile(cfg.models_root, spec.sub, spec.session_id, spec.run);
    spm_mat = fullfile(model_dir, 'SPM.mat');

    if exist(model_dir, 'dir') == 7 && cfg.overwrite_existing
        rmdir(model_dir, 's');
    end
    ensure_dir(model_dir);

    matlabbatch = [];
    matlabbatch{1}.spm.stats.fmri_spec.dir = {model_dir};
    matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
    matlabbatch{1}.spm.stats.fmri_spec.timing.RT = cfg.tr;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = cfg.fmri_t;
    matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = cfg.fmri_t0;
    matlabbatch{1}.spm.stats.fmri_spec.sess.scans = scans;
    matlabbatch{1}.spm.stats.fmri_spec.sess.cond = conditions;
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.regress = regressors;
    matlabbatch{1}.spm.stats.fmri_spec.sess.multi_reg = {''};
    matlabbatch{1}.spm.stats.fmri_spec.sess.hpf = cfg.hpf;
    matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
    matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
    matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
    matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
    matlabbatch{1}.spm.stats.fmri_spec.mthresh = 0;
    matlabbatch{1}.spm.stats.fmri_spec.mask = {mask_file};
    matlabbatch{1}.spm.stats.fmri_spec.cvi = 'AR(1)';

    if cfg.do_estimate
        matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat};
        matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
        matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
    end

    spm_jobman('run', matlabbatch);

    if cfg.do_estimate
        contrast_lookup = apply_condition_contrasts(spm_mat);
    else
        for i = 1:numel(contrast_lookup)
            contrast_lookup(i).note = 'specification_only';
        end
    end

    run_rows = initialize_run_result_row( ...
        spec, valuation_summary, height(collapsed_target_rows), spm_mat, contrast_lookup, selection_note, cfg.do_estimate);
catch ME
    if isempty(valuation_summary)
        valuation_summary = summarize_selected_valuation_rows(selected_target_rows);
    end
    run_rows = initialize_run_failure_row( ...
        spec, valuation_summary, height(collapsed_target_rows), contrast_lookup, spm_mat, simplify_error_message(ME));
end
end


function [target_rows, miss_trial_ids] = select_nonmiss_target_rows(events)
miss_mask = ismember(events.trial_type, ["miss", "invalid", "nonresponse", "non_response"]);
miss_trial_ids = unique(double(events.trial_id(miss_mask)));

valuation_rows = events(events.trial_type == "valuation", :);
target_mask = ~ismember(double(valuation_rows.trial_id), miss_trial_ids);
target_rows = valuation_rows(target_mask, :);
if ~isempty(target_rows)
    target_rows = sortrows(target_rows, {'onset', 'trial_id'});
end
end


function [selected_target_rows, collapsed_target_rows, selection_note] = select_target_rows_for_run(target_rows, pilot_only, pilot_max_target_trials)
if ~pilot_only
    selected_target_rows = target_rows;
    collapsed_target_rows = target_rows([],:);
    selection_note = 'ok';
    return;
end

n_keep = min(pilot_max_target_trials, height(target_rows));
selected_target_rows = target_rows(1:n_keep, :);
collapsed_target_rows = target_rows((n_keep + 1):end, :);

if isempty(collapsed_target_rows)
    selection_note = sprintf('ok_pilot_first_%d_targets', n_keep);
else
    selection_note = sprintf('ok_pilot_first_%d_targets_remaining_%d_collapsed_to_other_valuation_trials', ...
        n_keep, height(collapsed_target_rows));
end
end


function conditions = build_condition_level_conditions(events, selected_target_rows, collapsed_target_rows)
conditions = empty_condition_struct();

rating_values = round(sanitize_numeric_column(selected_target_rows.rating_value));
assigned_mask = false(height(selected_target_rows), 1);

for rating_level = 1:4
    rating_mask = isfinite(rating_values) & rating_values == rating_level;
    assigned_mask = assigned_mask | rating_mask;
    conditions = append_condition_if_present(conditions, sprintf('rating_%d', rating_level), ...
        selected_target_rows.onset(rating_mask), selected_target_rows.duration(rating_mask));
end

other_selected_mask = ~assigned_mask;
conditions = append_condition_if_present(conditions, 'other_selected_valuation', ...
    selected_target_rows.onset(other_selected_mask), selected_target_rows.duration(other_selected_mask));

if ~isempty(collapsed_target_rows)
    conditions(end + 1, 1) = make_condition( ... %#ok<AGROW>
        'other_valuation_trials', collapsed_target_rows.onset, collapsed_target_rows.duration);
end

conditions = append_condition_if_present(conditions, 'rating', ...
    events.onset(events.trial_type == "rating"), events.duration(events.trial_type == "rating"));
conditions = append_condition_if_present(conditions, 'response', ...
    events.onset(events.trial_type == "response"), events.duration(events.trial_type == "response"));
conditions = append_condition_if_present(conditions, 'feedback', ...
    events.onset(events.trial_type == "feedback"), events.duration(events.trial_type == "feedback"));

miss_mask = events.trial_type == "miss";
conditions = append_condition_if_present(conditions, 'miss', ...
    events.onset(miss_mask), events.duration(miss_mask));

if ~any(arrayfun(@(x) startsWith(string(x.name), "rating_"), conditions))
    error('No rating_1/rating_2/rating_3/rating_4 conditions were created for this run.');
end
end


function conditions = append_condition_if_present(conditions, name, onset_values, duration_values)
if isempty(onset_values)
    return;
end
conditions(end + 1, 1) = make_condition(name, onset_values, duration_values); %#ok<AGROW>
end


function condition = make_condition(name, onset_values, duration_values)
condition = struct();
condition.name = name;
condition.onset = double(onset_values(:)');
condition.duration = double(duration_values(:)');
condition.tmod = 0;
condition.pmod = struct('name', {}, 'param', {}, 'poly', {});
condition.orth = 0;
end


function conditions = empty_condition_struct()
conditions = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
end


function regressors = build_nuisance_regressors(confounds_file, motion_names, extra_nuisance_names, fd_threshold, head_radius_mm)
confounds = readtable(confounds_file, 'FileType', 'text', 'Delimiter', '\t', 'TreatAsEmpty', {'n/a', 'NA'});
required_names = [motion_names(:)' extra_nuisance_names(:)'];
must_have_columns(confounds, required_names);

motion = zeros(height(confounds), numel(motion_names));
for i = 1:numel(motion_names)
    motion(:, i) = sanitize_numeric_column(confounds.(motion_names{i}));
end

fd = compute_fd_from_motion(motion, head_radius_mm);
spike_idx = find(fd > fd_threshold);

regressors = struct('name', {}, 'val', {});
for i = 1:numel(motion_names)
    regressors(end + 1).name = motion_names{i}; %#ok<AGROW>
    regressors(end).val = motion(:, i);
end

for i = 1:numel(extra_nuisance_names)
    regressors(end + 1).name = extra_nuisance_names{i}; %#ok<AGROW>
    regressors(end).val = sanitize_numeric_column(confounds.(extra_nuisance_names{i}));
end

for i = 1:numel(spike_idx)
    spike = zeros(size(motion, 1), 1);
    spike(spike_idx(i)) = 1;
    regressors(end + 1).name = sprintf('fd_spike_%04d', i); %#ok<AGROW>
    regressors(end).val = spike;
end
end


function values = sanitize_numeric_column(raw_values)
if iscell(raw_values)
    raw_values = string(raw_values);
end
if isstring(raw_values) || ischar(raw_values)
    raw_values = str2double(string(raw_values));
end
values = double(raw_values);
values(~isfinite(values)) = 0;
end


function fd = compute_fd_from_motion(motion, head_radius_mm)
dm = [zeros(1, size(motion, 2)); diff(motion, 1, 1)];
fd = sum(abs(dm(:, 1:3)), 2) + head_radius_mm * sum(abs(dm(:, 4:6)), 2);
fd(~isfinite(fd)) = 0;
end


function contrast_lookup = apply_condition_contrasts(spm_mat)
contrast_lookup = default_condition_contrast_specs();

if exist(spm_mat, 'file') ~= 2
    for i = 1:numel(contrast_lookup)
        contrast_lookup(i).note = 'spm_mat_missing';
    end
    return;
end

S = load(spm_mat, 'SPM');
if ~isfield(S, 'SPM') || ~isfield(S.SPM, 'xX') || ~isfield(S.SPM.xX, 'name')
    for i = 1:numel(contrast_lookup)
        contrast_lookup(i).note = 'spm_structure_incomplete_after_estimation';
    end
    return;
end

design_names = string(S.SPM.xX.name(:));
matlabbatch = [];
n_consess = 0;

for i = 1:numel(contrast_lookup)
    [weights, note] = build_condition_contrast_weights( ...
        design_names, contrast_lookup(i).positive_condition, contrast_lookup(i).negative_condition);

    if isempty(weights)
        contrast_lookup(i).note = note;
        continue;
    end

    n_consess = n_consess + 1;
    matlabbatch{1}.spm.stats.con.consess{n_consess}.tcon.name = contrast_lookup(i).name;
    matlabbatch{1}.spm.stats.con.consess{n_consess}.tcon.weights = weights;
    matlabbatch{1}.spm.stats.con.consess{n_consess}.tcon.sessrep = 'none';
    contrast_lookup(i).note = 'pending_lookup';
end

if n_consess == 0
    return;
end

matlabbatch{1}.spm.stats.con.spmmat = {spm_mat};
matlabbatch{1}.spm.stats.con.delete = 1;
spm_jobman('run', matlabbatch);

S = load(spm_mat, 'SPM');
if ~isfield(S.SPM, 'xCon') || isempty(S.SPM.xCon)
    for i = 1:numel(contrast_lookup)
        if strcmp(contrast_lookup(i).note, 'pending_lookup')
            contrast_lookup(i).note = 'contrast_outputs_missing_after_definition';
        end
    end
    return;
end

xcon_names = string({S.SPM.xCon.name});
model_dir = fileparts(spm_mat);

for i = 1:numel(contrast_lookup)
    if ~strcmp(contrast_lookup(i).note, 'pending_lookup')
        continue;
    end

    idx = find(xcon_names == string(contrast_lookup(i).name), 1, 'first');
    if isempty(idx)
        contrast_lookup(i).note = 'contrast_not_found_in_spm_xCon';
        continue;
    end

    if isfield(S.SPM.xCon(idx), 'Vcon') && isfield(S.SPM.xCon(idx).Vcon, 'fname')
        contrast_lookup(i).con_file = pick_existing_file(fullfile(model_dir, S.SPM.xCon(idx).Vcon.fname));
    end
    if isfield(S.SPM.xCon(idx), 'Vspm') && isfield(S.SPM.xCon(idx).Vspm, 'fname')
        contrast_lookup(i).spmT_file = pick_existing_file(fullfile(model_dir, S.SPM.xCon(idx).Vspm.fname));
    end

    contrast_lookup(i).available = ~isempty(contrast_lookup(i).con_file) && ~isempty(contrast_lookup(i).spmT_file);
    if contrast_lookup(i).available
        contrast_lookup(i).note = 'ok';
    else
        contrast_lookup(i).note = 'contrast_defined_but_output_missing';
    end
end
end


function [weights, note] = build_condition_contrast_weights(design_names, positive_condition, negative_condition)
weights = [];
note = 'unknown';

positive_name = "Sn(1) " + string(positive_condition) + "*bf(1)";
negative_name = "Sn(1) " + string(negative_condition) + "*bf(1)";

positive_idx = find(design_names == positive_name, 1, 'first');
negative_idx = find(design_names == negative_name, 1, 'first');

if isempty(positive_idx) && isempty(negative_idx)
    note = sprintf('missing_%s_and_%s', positive_condition, negative_condition);
    return;
end
if isempty(positive_idx)
    note = sprintf('missing_%s', positive_condition);
    return;
end
if isempty(negative_idx)
    note = sprintf('missing_%s', negative_condition);
    return;
end

weights = zeros(1, numel(design_names));
weights(positive_idx) = 1;
weights(negative_idx) = -1;
note = 'ok';
end


function contrast_lookup = default_condition_contrast_specs()
contrast_lookup = struct( ...
    'name', {'rating1_gt_rating4', 'rating4_gt_rating1'}, ...
    'positive_condition', {'rating_1', 'rating_4'}, ...
    'negative_condition', {'rating_4', 'rating_1'}, ...
    'con_file', {'', ''}, ...
    'spmT_file', {'', ''}, ...
    'available', {false, false}, ...
    'note', {'not_run', 'not_run'});
end


function row = initialize_run_result_row(spec, valuation_summary, n_collapsed_valuation_trials, spm_mat, contrast_lookup, selection_note, do_estimate)
row = make_empty_result_row();
row.sub = spec.sub;
row.session_id = spec.session_id;
row.run = spec.run;
row.n_selected_valuation_trials = valuation_summary.n_selected_valuation_trials;
row.n_collapsed_valuation_trials = double(n_collapsed_valuation_trials);
row.n_rating1 = valuation_summary.n_rating1;
row.n_rating2 = valuation_summary.n_rating2;
row.n_rating3 = valuation_summary.n_rating3;
row.n_rating4 = valuation_summary.n_rating4;
row.n_other_selected_valuation = valuation_summary.n_other_selected_valuation;
row.spm_mat = pick_existing_file(spm_mat);
row.model_estimated_successfully = do_estimate && ~isempty(row.spm_mat);

row = assign_contrast_fields(row, contrast_lookup);
row.note = compose_run_note(selection_note, contrast_lookup, do_estimate, row.model_estimated_successfully);
end


function row = initialize_run_failure_row(spec, valuation_summary, n_collapsed_valuation_trials, contrast_lookup, spm_mat, note)
row = make_empty_result_row();
row.sub = spec.sub;
row.session_id = spec.session_id;
row.run = spec.run;
row.n_selected_valuation_trials = valuation_summary.n_selected_valuation_trials;
row.n_collapsed_valuation_trials = double(n_collapsed_valuation_trials);
row.n_rating1 = valuation_summary.n_rating1;
row.n_rating2 = valuation_summary.n_rating2;
row.n_rating3 = valuation_summary.n_rating3;
row.n_rating4 = valuation_summary.n_rating4;
row.n_other_selected_valuation = valuation_summary.n_other_selected_valuation;
row.spm_mat = pick_existing_file(spm_mat);
row = assign_contrast_fields(row, contrast_lookup);
row.note = ['run_level_failure: ' note];
end


function row = assign_contrast_fields(row, contrast_lookup)
for i = 1:numel(contrast_lookup)
    switch contrast_lookup(i).name
        case 'rating1_gt_rating4'
            row.con_1gt4 = contrast_lookup(i).con_file;
            row.spmT_1gt4 = contrast_lookup(i).spmT_file;
            row.contrast_rating1_gt_rating4_available = contrast_lookup(i).available;
            row.contrast_rating1_gt_rating4_note = contrast_lookup(i).note;
        case 'rating4_gt_rating1'
            row.con_4gt1 = contrast_lookup(i).con_file;
            row.spmT_4gt1 = contrast_lookup(i).spmT_file;
            row.contrast_rating4_gt_rating1_available = contrast_lookup(i).available;
            row.contrast_rating4_gt_rating1_note = contrast_lookup(i).note;
    end
end
end


function note = compose_run_note(selection_note, contrast_lookup, do_estimate, model_estimated_successfully)
parts = {};

if ~do_estimate
    parts{end + 1} = 'specification_only'; %#ok<AGROW>
elseif ~model_estimated_successfully
    parts{end + 1} = 'spm_mat_missing_after_estimation'; %#ok<AGROW>
end

if ~strcmp(selection_note, 'ok')
    parts{end + 1} = selection_note; %#ok<AGROW>
end

for i = 1:numel(contrast_lookup)
    if strcmp(contrast_lookup(i).note, 'ok') || strcmp(contrast_lookup(i).note, 'specification_only')
        continue;
    end
    parts{end + 1} = sprintf('%s:%s', contrast_lookup(i).name, contrast_lookup(i).note); %#ok<AGROW>
end

if isempty(parts)
    note = 'ok';
else
    note = strjoin(parts, '; ');
end
end


function summary = summarize_selected_valuation_rows(selected_target_rows)
summary = make_empty_valuation_summary();

if isempty(selected_target_rows)
    return;
end

summary.n_selected_valuation_trials = height(selected_target_rows);
rating_values = round(sanitize_numeric_column(selected_target_rows.rating_value));

summary.n_rating1 = nnz(isfinite(rating_values) & rating_values == 1);
summary.n_rating2 = nnz(isfinite(rating_values) & rating_values == 2);
summary.n_rating3 = nnz(isfinite(rating_values) & rating_values == 3);
summary.n_rating4 = nnz(isfinite(rating_values) & rating_values == 4);
summary.n_other_selected_valuation = summary.n_selected_valuation_trials - ...
    summary.n_rating1 - summary.n_rating2 - summary.n_rating3 - summary.n_rating4;
end


function summary = make_empty_valuation_summary()
summary = struct();
summary.n_selected_valuation_trials = 0;
summary.n_rating1 = 0;
summary.n_rating2 = 0;
summary.n_rating3 = 0;
summary.n_rating4 = 0;
summary.n_other_selected_valuation = 0;
end


function row = make_empty_result_row()
row = struct();
row.sub = '';
row.session_id = '';
row.run = '';
row.n_selected_valuation_trials = 0;
row.n_collapsed_valuation_trials = 0;
row.n_rating1 = 0;
row.n_rating2 = 0;
row.n_rating3 = 0;
row.n_rating4 = 0;
row.n_other_selected_valuation = 0;
row.spm_mat = '';
row.model_estimated_successfully = false;
row.con_1gt4 = '';
row.spmT_1gt4 = '';
row.contrast_rating1_gt_rating4_available = false;
row.contrast_rating1_gt_rating4_note = '';
row.con_4gt1 = '';
row.spmT_4gt1 = '';
row.contrast_rating4_gt_rating1_available = false;
row.contrast_rating4_gt_rating1_note = '';
row.note = '';
end


function results = force_nonempty_results(results)
if isempty(results)
    results = make_empty_result_row();
    results.note = 'no_rows_generated';
end
end


function results = concat_result_cells(results_cells)
results = struct([]);
for i = 1:numel(results_cells)
    if isempty(results_cells{i})
        continue;
    end
    results = append_rows(results, results_cells{i});
end
end


function rows = append_rows(rows, new_rows)
if isempty(new_rows)
    return;
end

if ~isstruct(new_rows)
    error('append_rows expects a struct array.');
end

if isempty(rows)
    rows = new_rows(:);
    return;
end

template = rows(1);
for i = 1:numel(new_rows)
    rows(end + 1, 1) = orderfields(new_rows(i), template); %#ok<AGROW>
end
end


function stimulus_id = get_stimulus_id(tbl_row)
stimulus_id = get_text_value(tbl_row, 'stimulus_id');
if isempty(stimulus_id)
    stimulus_id = get_text_value(tbl_row, 'image_id');
end
end


function image_identity_key = get_image_identity_key(tbl_row, stimulus_id)
image_identity_key = get_text_value(tbl_row, 'image_identity_key');
if ~isempty(image_identity_key)
    return;
end

numeric_id = str2double(stimulus_id);
if ~isnan(numeric_id)
    image_identity_key = sprintf('img-%04d', round(numeric_id));
else
    image_identity_key = '';
end
end


function text_value = get_text_value(tbl_row, var_name)
text_value = '';
if ~ismember(var_name, tbl_row.Properties.VariableNames)
    return;
end

value = tbl_row.(var_name)(1);
if iscell(value)
    value = value{1};
end
if isstring(value)
    value = char(value);
end

if isnumeric(value)
    if isempty(value) || ~isfinite(value)
        text_value = '';
    else
        text_value = sprintf('%g', value);
    end
elseif ischar(value)
    text_value = strtrim(value);
end
end


function value = get_numeric_value(tbl_row, var_name)
value = NaN;
if ~ismember(var_name, tbl_row.Properties.VariableNames)
    return;
end

raw = tbl_row.(var_name)(1);
if iscell(raw)
    raw = raw{1};
end
if isstring(raw) || ischar(raw)
    raw = str2double(string(raw));
end
if isnumeric(raw) && ~isempty(raw) && isfinite(raw)
    value = double(raw);
end
end


function specs = discover_available_runs(events_root)
d = dir(fullfile(events_root, 'sub-*', 'ses-*', 'func', '*_events.tsv'));
specs = struct('sub', {}, 'session_id', {}, 'run', {}, 'events_file', {});

for i = 1:numel(d)
    if startsWith(d(i).name, '._')
        continue;
    end

    tokens = regexp(d(i).name, '^(sub-\d+)_(ses-\d+)_task-food_(run-\d+)_events\.tsv$', 'tokens', 'once');
    if isempty(tokens)
        continue;
    end

    specs(end + 1).sub = tokens{1}; %#ok<AGROW>
    specs(end).session_id = tokens{2};
    specs(end).run = tokens{3};
    specs(end).events_file = fullfile(d(i).folder, d(i).name);
end

if ~isempty(specs)
    sort_keys = arrayfun(@(x) sprintf('%s_%s_%s', x.sub, x.session_id, x.run), specs, 'UniformOutput', false);
    [~, order] = sort(sort_keys);
    specs = specs(order);
end
end


function selected = filter_run_specs(specs, subject_id, session_id, run_id)
selected = specs(false(size(specs)));
keep = false(numel(specs), 1);

for i = 1:numel(specs)
    keep(i) = matches_requested_token(specs(i).sub, subject_id) && ...
        matches_requested_token(specs(i).session_id, session_id) && ...
        matches_requested_token(specs(i).run, run_id);
end

selected = specs(keep);
end


function tf = matches_requested_token(value, requested)
requested = cellstr(string(requested));
tf = any(strcmpi(requested, 'ALL')) || any(strcmp(value, requested));
end


function [bold_file, confounds_file, events_file, mask_file] = resolve_input_files(input_root, cache_root, subject_id, session_id, run_id)
func_root = fullfile(input_root, subject_id, session_id, 'func');
bold_stem = sprintf('%s_%s_task-food_%s_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold', subject_id, session_id, run_id);

bold_file = pick_first_existing({
    fullfile(func_root, bold_stem, [bold_stem '.nii']), ...
    fullfile(func_root, [bold_stem '.nii']), ...
    fullfile(func_root, [bold_stem '.nii.gz'])});
if isempty(bold_file)
    error('BOLD file not found for %s %s %s', subject_id, session_id, run_id);
end
bold_file = prepare_nifti_if_needed(bold_file, fullfile(cache_root, 'nifti'));

confounds_file = fullfile(func_root, sprintf('%s_%s_task-food_%s_desc-confounds_timeseries.tsv', subject_id, session_id, run_id));
must_exist(confounds_file, 'Confounds file');

events_file = fullfile(input_root, 'events', subject_id, session_id, 'func', ...
    sprintf('%s_%s_task-food_%s_events.tsv', subject_id, session_id, run_id));
must_exist(events_file, 'Events file');

anat_mask = pick_first_existing({
    fullfile(input_root, subject_id, 'anat', sprintf('%s_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii', subject_id)), ...
    fullfile(input_root, subject_id, 'anat', sprintf('%s_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz', subject_id))});

run_mask = pick_first_existing({
    fullfile(func_root, sprintf('%s_%s_task-food_%s_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii', subject_id, session_id, run_id)), ...
    fullfile(func_root, sprintf('%s_%s_task-food_%s_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz', subject_id, session_id, run_id)), ...
    fullfile(func_root, sprintf('%s_%s_task-food_%s_desc-brain_mask.nii', subject_id, session_id, run_id)), ...
    fullfile(func_root, sprintf('%s_%s_task-food_%s_desc-brain_mask.nii.gz', subject_id, session_id, run_id))});

mask_source = anat_mask;
if isempty(mask_source)
    mask_source = run_mask;
end
if isempty(mask_source)
    error('No subject-level or run-level brain mask found for %s %s %s', subject_id, session_id, run_id);
end

mask_file = prepare_nifti_if_needed(mask_source, fullfile(cache_root, 'masks'));
end


function nii_file = prepare_nifti_if_needed(source_file, cache_dir)
if endsWith(source_file, '.nii', 'IgnoreCase', true)
    nii_file = source_file;
    return;
end

if ~endsWith(source_file, '.nii.gz', 'IgnoreCase', true)
    error('Expected .nii or .nii.gz file, got: %s', source_file);
end

ensure_dir(cache_dir);
if exist(cache_dir, 'dir') == 7
    [~, name] = fileparts(source_file);
    [~, base_name] = fileparts(name);
    nii_file = fullfile(cache_dir, [base_name '.nii']);
    if exist(nii_file, 'file') == 2
        return;
    end
end
gunzip(source_file, cache_dir);
[~, name] = fileparts(source_file);
[~, base_name] = fileparts(name);
nii_file = fullfile(cache_dir, [base_name '.nii']);
must_exist(nii_file, 'Cached NIfTI file');
end


function smoothed_bold_file = prepare_smoothed_bold_if_needed(bold_file, cache_root, subject_id, session_id, run_id, smooth_fwhm_mm)
if smooth_fwhm_mm <= 0
    smoothed_bold_file = bold_file;
    return;
end

[~, base_name, ext] = fileparts(bold_file);
smooth_dir = fullfile(cache_root, 'smoothed', subject_id, session_id, run_id);
ensure_dir(smooth_dir);

smooth_label = format_smoothing_label(smooth_fwhm_mm);
smoothed_bold_file = fullfile(smooth_dir, sprintf('s%smm_%s%s', smooth_label, base_name, ext));

if is_complete_smoothed_file(smoothed_bold_file, bold_file)
    return;
end

if exist(smoothed_bold_file, 'file') == 2
    delete(smoothed_bold_file);
end

input_scans = build_scan_list(bold_file);
for i = 1:numel(input_scans)
    output_scan = sprintf('%s,%d', smoothed_bold_file, i);
    spm_smooth(input_scans{i}, output_scan, [smooth_fwhm_mm smooth_fwhm_mm smooth_fwhm_mm], 0);
end

if ~is_complete_smoothed_file(smoothed_bold_file, bold_file)
    error('Smoothed BOLD file is incomplete after smoothing: %s', smoothed_bold_file);
end
end


function tf = is_complete_smoothed_file(smoothed_bold_file, reference_bold_file)
tf = exist(smoothed_bold_file, 'file') == 2;
if ~tf
    return;
end

try
    ref_vols = spm_vol(reference_bold_file);
    smoothed_vols = spm_vol(smoothed_bold_file);
    tf = numel(ref_vols) == numel(smoothed_vols);
catch
    tf = false;
end
end


function text = format_smoothing_label(smooth_fwhm_mm)
text = regexprep(num2str(double(smooth_fwhm_mm), '%.3f'), '0+$', '');
text = regexprep(text, '\.$', '');
text = strrep(text, '.', 'p');
end


function scans = build_scan_list(bold_file)
V = spm_vol(bold_file);
scans = cell(numel(V), 1);
for i = 1:numel(V)
    scans{i} = sprintf('%s,%d', bold_file, i);
end
end


function out = pick_first_existing(candidates)
out = '';
for i = 1:numel(candidates)
    if exist(candidates{i}, 'file') == 2
        out = candidates{i};
        return;
    end
end
end


function pathstr = pick_existing_file(pathstr)
if isempty(pathstr) || exist(pathstr, 'file') ~= 2
    pathstr = '';
end
end


function ensure_dir(pathstr)
if exist(pathstr, 'dir') ~= 7
    mkdir(pathstr);
end
end


function must_exist(pathstr, label)
if exist(pathstr, 'file') ~= 2
    error('%s not found: %s', label, pathstr);
end
end


function must_have_columns(tbl, names)
missing = names(~ismember(names, tbl.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required column(s): %s', strjoin(missing, ', '));
end
end


function text = simplify_error_message(ME)
text = regexprep(ME.message, '\s+', ' ');
end
