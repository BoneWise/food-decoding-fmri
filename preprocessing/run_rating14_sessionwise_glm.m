% Session-wise run-wise rating1/rating4 first-level GLM for the food task.
% This script keeps one SPM model per subject x BIDS session, with each run
% represented as one SPM session block. It produces run-wise rating condition
% betas for the primary 3-fold leave-one-session-out searchlight MVPA.

clearvars -except subject_id session_id run_id do_estimate overwrite_existing keep_only_rating_betas spm_dir max_parallel_subjects input_root output_root
clc;

% -------------------- edit here --------------------
if ~exist('subject_id', 'var'), subject_id = 'ALL'; end
if ~exist('session_id', 'var'), session_id = 'ALL'; end
if ~exist('run_id', 'var'), run_id = 'ALL'; end
if ~exist('do_estimate', 'var'), do_estimate = true; end
if ~exist('overwrite_existing', 'var'), overwrite_existing = true; end
if ~exist('keep_only_rating_betas', 'var'), keep_only_rating_betas = true; end
if ~exist('spm_dir', 'var'), spm_dir = ''; end
if ~exist('max_parallel_subjects', 'var'), max_parallel_subjects = 3; end
% --------------------------------------------------

if ~exist('input_root', 'var') || isempty(input_root)
    input_root = getenv('FOODPREF_FMRIPREP_ROOT');
end
if isempty(input_root)
    error('Set input_root or FOODPREF_FMRIPREP_ROOT to the fMRIPrep derivatives directory.');
end
events_root = fullfile(input_root, 'events');
script_dir = fileparts(mfilename('fullpath'));
if ~exist('output_root', 'var') || isempty(output_root)
    output_root = fullfile(script_dir, '..', 'data', 'derived', 'sessionwise');
end
models_root = fullfile(output_root, 'models');
cache_root = fullfile(output_root, 'cache');
index_csv = fullfile(output_root, 'beta_index.csv');
contrast_csv = fullfile(output_root, 'contrast_index.csv');
summary_csv = fullfile(output_root, 'runwise_glm_summary.csv');

tr = 0.8;
fmri_t = 60;
fmri_t0 = 30;
hpf = 128;
fd_threshold = 0.5;
head_radius_mm = 50;
motion_names = {'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'};
n_acompcor = 6;

ensure_dir(output_root);
ensure_dir(models_root);
ensure_dir(cache_root);

[resolved_spm_dir, spm_release] = setup_spm25(spm_dir);
fprintf('SPM release: %s\n', spm_release);
fprintf('SPM dir: %s\n', resolved_spm_dir);

all_specs = discover_available_runs(events_root);
if isempty(all_specs)
    error('No event files were discovered under %s', events_root);
end

selected_specs = filter_run_specs(all_specs, subject_id, session_id, run_id);
if isempty(selected_specs)
    error('No runs matched subject=%s session=%s run=%s', char(string(subject_id)), char(string(session_id)), char(string(run_id)));
end

fprintf('Selected runs: %d\n', numel(selected_specs));

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
cfg.n_acompcor = n_acompcor;
cfg.do_estimate = do_estimate;
cfg.overwrite_existing = overwrite_existing;
cfg.keep_only_rating_betas = keep_only_rating_betas;
cfg.spm_dir = resolved_spm_dir;

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

beta_rows = concat_result_field(results_cells, 'beta_rows');
contrast_rows = concat_result_field(results_cells, 'contrast_rows');
summary_rows = concat_result_field(results_cells, 'summary_rows');

beta_table = struct2table(force_nonempty_results(beta_rows, make_empty_beta_row()));
contrast_table = struct2table(force_nonempty_results(contrast_rows, make_empty_contrast_row()));
summary_table = struct2table(force_nonempty_results(summary_rows, make_empty_summary_row()));

writetable(beta_table, index_csv);
writetable(contrast_table, contrast_csv);
writetable(summary_table, summary_csv);

fprintf('\nDone.\n');
fprintf('Beta index: %s\n', index_csv);
fprintf('Contrast index: %s\n', contrast_csv);
fprintf('Run-wise summary: %s\n', summary_csv);
fprintf('Beta rows written: %d\n', height(beta_table));
fprintf('Estimated rating betas: %d\n', nnz(beta_table.estimated_successfully));


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


function subject_rows = run_subject_block(subject_specs, cfg)
setup_spm25(cfg.spm_dir);
subject_rows = make_empty_session_result();

fprintf('\n=== %s (%d runs) ===\n', subject_specs(1).sub, numel(subject_specs));

session_ids = unique({subject_specs.session_id}, 'stable');
for i = 1:numel(session_ids)
    session_specs = subject_specs(strcmp({subject_specs.session_id}, session_ids{i}));
    fprintf('\n--- %s %s (%d runs) ---\n', session_specs(1).sub, session_specs(1).session_id, numel(session_specs));
    session_result = run_single_session_rating14_glm(session_specs, cfg);
    subject_rows.beta_rows = append_rows(subject_rows.beta_rows, session_result.beta_rows);
    subject_rows.contrast_rows = append_rows(subject_rows.contrast_rows, session_result.contrast_rows);
    subject_rows.summary_rows = append_rows(subject_rows.summary_rows, session_result.summary_rows);
end
end


function session_result = make_empty_session_result()
session_result = struct();
session_result.beta_rows = struct([]);
session_result.contrast_rows = struct([]);
session_result.summary_rows = struct([]);
end


function info = make_empty_run_info()
info = struct();
info.spec = struct('sub', '', 'session_id', '', 'run', '', 'events_file', '');
info.spm_session_index = NaN;
info.scans = {};
info.conditions = empty_condition_struct();
info.regressors = struct('name', {}, 'val', {});
info.run_qc = struct('spike_count', NaN, 'spike_fraction', NaN, 'n_regressors', NaN);
info.condition_info = struct( ...
    'n_target_trials', NaN, ...
    'n_rating1_trials', NaN, ...
    'n_rating4_trials', NaN, ...
    'n_other_valuation_trials', NaN, ...
    'n_bad_events', NaN, ...
    'n_conditions', NaN);
end


function session_result = run_single_session_rating14_glm(session_specs, cfg)
session_result = make_empty_session_result();
spm_mat = '';
run_infos = repmat(make_empty_run_info(), numel(session_specs), 1);

try
    mask_file = '';

    for run_idx = 1:numel(session_specs)
        spec = session_specs(run_idx);
        fprintf('Preparing %s %s %s\n', spec.sub, spec.session_id, spec.run);

        [bold_file, confounds_file, events_file, this_mask_file] = resolve_input_files( ...
            cfg.input_root, cfg.cache_root, spec.sub, spec.session_id, spec.run);
        if isempty(mask_file)
            mask_file = this_mask_file;
        end

        events = readtable(events_file, 'FileType', 'text', 'Delimiter', '\t');
        must_have_columns(events, {'onset', 'duration', 'trial_id', 'trial_type', 'rating_value'});
        events.trial_type = lower(string(events.trial_type));

        [target_rows, ~] = select_nonmiss_target_rows(events);
        [conditions, condition_info] = build_rating14_run_conditions(events, target_rows);

        [regressors, run_qc] = build_confound_regressors( ...
            confounds_file, cfg.motion_names, cfg.fd_threshold, cfg.head_radius_mm, cfg.n_acompcor);

        run_infos(run_idx).spec = spec;
        run_infos(run_idx).spm_session_index = run_idx;
        run_infos(run_idx).scans = build_scan_list(bold_file);
        run_infos(run_idx).conditions = conditions;
        run_infos(run_idx).regressors = regressors;
        run_infos(run_idx).run_qc = run_qc;
        run_infos(run_idx).condition_info = condition_info;

        fprintf('  rating1=%d rating4=%d other=%d conditions=%d\n', ...
            condition_info.n_rating1_trials, condition_info.n_rating4_trials, ...
            condition_info.n_other_valuation_trials, condition_info.n_conditions);
    end

    model_dir = fullfile(cfg.models_root, session_specs(1).sub, session_specs(1).session_id);
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
    for run_idx = 1:numel(run_infos)
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).scans = run_infos(run_idx).scans;
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).cond = run_infos(run_idx).conditions;
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).multi = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).regress = run_infos(run_idx).regressors;
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).multi_reg = {''};
        matlabbatch{1}.spm.stats.fmri_spec.sess(run_idx).hpf = cfg.hpf;
    end
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

    beta_lookup = lookup_rating14_betas(spm_mat, run_infos, cfg.do_estimate);
    contrast_lookup = create_rating14_contrasts(spm_mat, run_infos, beta_lookup, cfg.do_estimate);
    if cfg.keep_only_rating_betas
        cleanup_non_rating_betas(spm_mat, beta_lookup, cfg.do_estimate);
    end

    session_result.beta_rows = initialize_beta_rows(run_infos, spm_mat, model_dir, beta_lookup);
    session_result.contrast_rows = initialize_contrast_rows(run_infos, spm_mat, contrast_lookup);
    session_result.summary_rows = initialize_summary_rows(run_infos, spm_mat, model_dir, beta_lookup, contrast_lookup);
catch ME
    session_result = initialize_session_failure_result(session_specs, spm_mat, simplify_error_message(ME));
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


function [conditions, condition_info] = build_rating14_run_conditions(events, target_rows)
conditions = empty_condition_struct();
condition_info = struct();
condition_info.n_target_trials = height(target_rows);
condition_info.n_rating1_trials = 0;
condition_info.n_rating4_trials = 0;
condition_info.n_other_valuation_trials = 0;
condition_info.n_bad_events = 0;
condition_info.n_conditions = 0;

rating_values = [];
if ~isempty(target_rows)
    rating_values = numeric_column_preserve_nan(target_rows.rating_value);

    rating1_mask = rating_values == 1;
    rating4_mask = rating_values == 4;
    other_mask = ~rating1_mask & ~rating4_mask;

    condition_info.n_rating1_trials = nnz(rating1_mask);
    condition_info.n_rating4_trials = nnz(rating4_mask);
    condition_info.n_other_valuation_trials = nnz(other_mask);

    if any(rating1_mask)
        conditions(end + 1, 1) = make_condition( ... %#ok<AGROW>
            'rating1_valuation', target_rows.onset(rating1_mask), target_rows.duration(rating1_mask));
    end
    if any(rating4_mask)
        conditions(end + 1, 1) = make_condition( ... %#ok<AGROW>
            'rating4_valuation', target_rows.onset(rating4_mask), target_rows.duration(rating4_mask));
    end
    if any(other_mask)
        conditions(end + 1, 1) = make_condition( ... %#ok<AGROW>
            'other_valuation', target_rows.onset(other_mask), target_rows.duration(other_mask));
    end
end

for cond_name = ["rating", "response", "feedback"]
    mask = events.trial_type == cond_name;
    if ~any(mask), continue; end
    cond = make_condition(char(cond_name), events.onset(mask), events.duration(mask));
    conditions(end + 1, 1) = cond; %#ok<AGROW>
end

bad_mask = ismember(events.trial_type, ["miss", "invalid", "nonresponse", "non_response"]);
condition_info.n_bad_events = nnz(bad_mask);
if any(bad_mask)
    cond = make_condition('bad_or_missed', events.onset(bad_mask), events.duration(bad_mask));
    conditions(end + 1, 1) = cond; %#ok<AGROW>
end

if isempty(conditions)
    error('No conditions were created for this run.');
end

condition_info.n_conditions = numel(conditions);
end


function condition = attach_pmods(condition, event_rows, pmods_list)
for p = 1:numel(pmods_list)
    pmod_name = pmods_list{p};
    if ~ismember(pmod_name, event_rows.Properties.VariableNames)
        continue;
    end
    raw = event_rows.(pmod_name);
    if iscell(raw), raw = cellfun(@(x) double(string(x)), raw); end
    if isstring(raw) || ischar(raw), raw = str2double(string(raw)); end
    raw = double(raw);
    if all(~isfinite(raw))
        continue;
    end
    vals = raw;
    vals(~isfinite(vals)) = 0;
    if std(vals) < eps
        continue;
    end
    vals = (vals - mean(vals)) / std(vals);
    k = numel(condition.pmod) + 1;
    condition.pmod(k).name = pmod_name;
    condition.pmod(k).param = double(vals(:)');
    condition.pmod(k).poly = 1;
end
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


function [regressors, run_qc] = build_confound_regressors(confounds_file, motion_names, fd_threshold, head_radius_mm, n_acompcor)
confounds = readtable(confounds_file, 'FileType', 'text', 'Delimiter', '\t', 'TreatAsEmpty', {'n/a', 'NA'});
must_have_columns(confounds, motion_names);
n_vols = height(confounds);
regressors = struct('name', {}, 'val', {});
spike_mask = false(n_vols, 1);

% --- 24 motion terms: 6 motion + 6 derivative + 6 squared + 6 derivative squared ---
motion = zeros(n_vols, numel(motion_names));
for i = 1:numel(motion_names)
    motion(:, i) = sanitize_numeric_column(confounds.(motion_names{i}));
end

motion_deriv = [zeros(1, size(motion, 2)); diff(motion, 1, 1)];
motion_sq = motion .^ 2;
motion_deriv_sq = motion_deriv .^ 2;

motion_24 = [motion, motion_deriv, motion_sq, motion_deriv_sq];
suffixes_24 = [motion_names, ...
    strcat(motion_names, '_derivative1'), ...
    strcat(motion_names, '_power2'), ...
    strcat(motion_names, '_derivative1_power2')];

for i = 1:size(motion_24, 2)
    regressors(end + 1).name = suffixes_24{i}; %#ok<AGROW>
    regressors(end).val = motion_24(:, i);
end

% --- aCompCor components ---
all_cols = confounds.Properties.VariableNames;
acompcor_cols = all_cols(startsWith(all_cols, 'a_comp_cor_'));
if ~isempty(acompcor_cols)
    nums = cellfun(@(c) sscanf(c, 'a_comp_cor_%d'), acompcor_cols);
    [~, sort_idx] = sort(nums);
    acompcor_cols = acompcor_cols(sort_idx);
    n_use = min(numel(acompcor_cols), n_acompcor);
    for i = 1:n_use
        regressors(end + 1).name = acompcor_cols{i}; %#ok<AGROW>
        regressors(end).val = sanitize_numeric_column(confounds.(acompcor_cols{i}));
    end
end

% --- std_dvars from TSV (continuous, if present) ---
if ismember('std_dvars', all_cols)
    regressors(end + 1).name = 'std_dvars'; %#ok<AGROW>
    regressors(end).val = sanitize_numeric_column(confounds.std_dvars);
end

% --- non_steady_state_outlier* as spike regressors ---
nss_cols = all_cols(startsWith(all_cols, 'non_steady_state_outlier'));
for i = 1:numel(nss_cols)
    vals = sanitize_numeric_column(confounds.(nss_cols{i}));
    if any(vals ~= 0)
        regressors(end + 1).name = nss_cols{i}; %#ok<AGROW>
        regressors(end).val = vals;
        spike_mask = spike_mask | (vals ~= 0);
    end
end

% --- motion_outlier* as spike regressors ---
mo_cols = all_cols(startsWith(all_cols, 'motion_outlier'));
for i = 1:numel(mo_cols)
    vals = sanitize_numeric_column(confounds.(mo_cols{i}));
    if any(vals ~= 0)
        regressors(end + 1).name = mo_cols{i}; %#ok<AGROW>
        regressors(end).val = vals;
        spike_mask = spike_mask | (vals ~= 0);
    end
end

% --- FD spikes (computed from motion) ---
fd = compute_fd_from_motion(motion, head_radius_mm);
spike_idx = find(fd > fd_threshold);
for i = 1:numel(spike_idx)
    spike = zeros(n_vols, 1);
    spike(spike_idx(i)) = 1;
    regressors(end + 1).name = sprintf('fd_spike_%04d', i); %#ok<AGROW>
    regressors(end).val = spike;
end
spike_mask(spike_idx) = true;

run_qc = struct();
run_qc.n_vols = n_vols;
run_qc.spike_mask = spike_mask;
run_qc.spike_count = nnz(spike_mask);
run_qc.spike_fraction = run_qc.spike_count / max(n_vols, 1);
run_qc.n_regressors = numel(regressors);
end


function trial_qc = build_trial_quality(selected_target_rows, run_qc, tr)
trial_qc = repmat(struct( ...
    'trial_spike_overlap', false, ...
    'trial_window_start_vol', NaN, ...
    'trial_window_end_vol', NaN, ...
    'trial_window_n_vols', NaN, ...
    'trial_window_spike_count', NaN, ...
    'trial_window_spike_fraction', NaN, ...
    'run_spike_count', run_qc.spike_count, ...
    'run_spike_fraction', run_qc.spike_fraction, ...
    'quality_label', 'usable', ...
    'quality_score', 3), height(selected_target_rows), 1);

for i = 1:height(selected_target_rows)
    onset = double(selected_target_rows.onset(i));
    duration = double(selected_target_rows.duration(i));
    start_vol = max(1, floor(max(0, onset - tr) / tr) + 1);
    end_vol = min(run_qc.n_vols, ceil((onset + duration + 4 * tr) / tr) + 1);
    if end_vol < start_vol
        end_vol = start_vol;
    end

    has_spike = false;
    spike_count = 0;
    window_n_vols = max(end_vol - start_vol + 1, 1);
    spike_fraction = 0;
    if run_qc.n_vols > 0
        spike_count = nnz(run_qc.spike_mask(start_vol:end_vol));
        has_spike = spike_count > 0;
        spike_fraction = spike_count / window_n_vols;
    end

    [quality_label, quality_score] = classify_quality_label(spike_count, spike_fraction, run_qc.spike_fraction);
    trial_qc(i).trial_spike_overlap = has_spike;
    trial_qc(i).trial_window_start_vol = start_vol;
    trial_qc(i).trial_window_end_vol = end_vol;
    trial_qc(i).trial_window_n_vols = window_n_vols;
    trial_qc(i).trial_window_spike_count = spike_count;
    trial_qc(i).trial_window_spike_fraction = spike_fraction;
    trial_qc(i).quality_label = quality_label;
    trial_qc(i).quality_score = quality_score;
end
end


function [quality_label, quality_score] = classify_quality_label(trial_spike_count, trial_spike_fraction, run_spike_fraction)
if trial_spike_count == 0 && run_spike_fraction <= 0.02
    quality_label = 'excellent';
    quality_score = 5;
elseif trial_spike_count == 0 && run_spike_fraction <= 0.05
    quality_label = 'good';
    quality_score = 4;
elseif trial_spike_count <= 1 && trial_spike_fraction <= 0.10 && run_spike_fraction <= 0.12
    quality_label = 'usable';
    quality_score = 3;
elseif trial_spike_count <= 2 && trial_spike_fraction <= 0.20 && run_spike_fraction <= 0.20
    quality_label = 'caution';
    quality_score = 2;
else
    quality_label = 'exclude';
    quality_score = 1;
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


function values = numeric_column_preserve_nan(raw_values)
if iscell(raw_values)
    raw_values = string(raw_values);
end
if isstring(raw_values) || ischar(raw_values)
    raw_values = str2double(string(raw_values));
end
values = double(raw_values);
values(~isfinite(values)) = NaN;
end


function fd = compute_fd_from_motion(motion, head_radius_mm)
dm = [zeros(1, size(motion, 2)); diff(motion, 1, 1)];
fd = sum(abs(dm(:, 1:3)), 2) + head_radius_mm * sum(abs(dm(:, 4:6)), 2);
fd(~isfinite(fd)) = 0;
end


function beta_lookup = lookup_rating14_betas(spm_mat, run_infos, do_estimate)
beta_lookup = repmat(make_empty_beta_lookup(), numel(run_infos), 1);
model_dir = fileparts(spm_mat);
design_names = strings(0, 1);
Vbeta = struct([]);
lookup_note = '';

if exist(spm_mat, 'file') ~= 2
    lookup_note = 'spm_mat_missing';
elseif ~do_estimate
    lookup_note = 'specification_only';
else
    S = load(spm_mat, 'SPM');
    if ~isfield(S, 'SPM') || ~isfield(S.SPM, 'xX') || ...
            ~isfield(S.SPM.xX, 'name') || ~isfield(S.SPM, 'Vbeta')
        lookup_note = 'spm_structure_incomplete_after_estimation';
    else
        design_names = string(S.SPM.xX.name(:));
        Vbeta = S.SPM.Vbeta;
    end
end

for i = 1:numel(run_infos)
    info = run_infos(i).condition_info;
    sess_idx = run_infos(i).spm_session_index;

    [beta_lookup(i).rating1_beta_file, beta_lookup(i).rating1_beta_index, ...
        beta_lookup(i).rating1_estimated_successfully, beta_lookup(i).rating1_note] = ...
        lookup_one_condition_beta(model_dir, design_names, Vbeta, sess_idx, ...
        'rating1_valuation', info.n_rating1_trials, lookup_note);

    [beta_lookup(i).rating4_beta_file, beta_lookup(i).rating4_beta_index, ...
        beta_lookup(i).rating4_estimated_successfully, beta_lookup(i).rating4_note] = ...
        lookup_one_condition_beta(model_dir, design_names, Vbeta, sess_idx, ...
        'rating4_valuation', info.n_rating4_trials, lookup_note);
end
end


function lookup = make_empty_beta_lookup()
lookup = struct();
lookup.rating1_beta_file = '';
lookup.rating1_beta_index = NaN;
lookup.rating1_estimated_successfully = false;
lookup.rating1_note = '';
lookup.rating4_beta_file = '';
lookup.rating4_beta_index = NaN;
lookup.rating4_estimated_successfully = false;
lookup.rating4_note = '';
end


function [beta_file, beta_index, estimated_successfully, note] = lookup_one_condition_beta(model_dir, design_names, Vbeta, sess_idx, condition_name, n_trials, lookup_note)
beta_file = '';
beta_index = NaN;
estimated_successfully = false;

if n_trials <= 0
    note = 'condition_absent_no_beta';
    return;
end

if ~isempty(lookup_note)
    note = lookup_note;
    return;
end

expected_name = "Sn(" + string(sess_idx) + ") " + string(condition_name) + "*bf(1)";
idx = find(design_names == expected_name, 1, 'first');
if isempty(idx) || idx > numel(Vbeta)
    note = 'rating_regressor_not_found_in_spm_xX_name';
    return;
end

beta_index = idx;
beta_path = fullfile(model_dir, Vbeta(idx).fname);
beta_file = pick_existing_file(beta_path);
estimated_successfully = exist(beta_path, 'file') == 2;
if estimated_successfully
    note = 'ok';
else
    note = 'beta_file_missing_after_estimation';
end
end


function contrast_lookup = create_rating14_contrasts(spm_mat, run_infos, beta_lookup, do_estimate)
contrast_lookup = repmat(make_empty_contrast_lookup(), numel(run_infos), 1);
for i = 1:numel(run_infos)
    contrast_lookup(i).rating4_minus_rating1_note = 'missing_rating1_or_rating4_beta';
    contrast_lookup(i).rating1_minus_rating4_note = 'missing_rating1_or_rating4_beta';
end

if ~do_estimate
    for i = 1:numel(run_infos)
        contrast_lookup(i).rating4_minus_rating1_note = 'specification_only';
        contrast_lookup(i).rating1_minus_rating4_note = 'specification_only';
    end
    return;
end

if exist(spm_mat, 'file') ~= 2
    for i = 1:numel(run_infos)
        contrast_lookup(i).rating4_minus_rating1_note = 'spm_mat_missing';
        contrast_lookup(i).rating1_minus_rating4_note = 'spm_mat_missing';
    end
    return;
end

S = load(spm_mat, 'SPM');
if ~isfield(S, 'SPM') || ~isfield(S.SPM, 'xX') || ~isfield(S.SPM.xX, 'name')
    for i = 1:numel(run_infos)
        contrast_lookup(i).rating4_minus_rating1_note = 'spm_structure_incomplete_for_contrasts';
        contrast_lookup(i).rating1_minus_rating4_note = 'spm_structure_incomplete_for_contrasts';
    end
    return;
end

n_cols = numel(S.SPM.xX.name);
contrast_defs = struct('run_idx', {}, 'direction', {}, 'name', {}, 'weights', {});
for i = 1:numel(run_infos)
    r1_idx = beta_lookup(i).rating1_beta_index;
    r4_idx = beta_lookup(i).rating4_beta_index;
    if ~isfinite(r1_idx) || ~isfinite(r4_idx)
        continue;
    end

    spec = run_infos(i).spec;
    prefix = sprintf('%s_%s_%s', spec.sub, spec.session_id, spec.run);

    weights = zeros(1, n_cols);
    weights(r4_idx) = 1;
    weights(r1_idx) = -1;
    contrast_defs(end + 1).run_idx = i; %#ok<AGROW>
    contrast_defs(end).direction = 'rating4_minus_rating1';
    contrast_defs(end).name = [prefix '_rating4_minus_rating1'];
    contrast_defs(end).weights = weights;

    weights = zeros(1, n_cols);
    weights(r1_idx) = 1;
    weights(r4_idx) = -1;
    contrast_defs(end + 1).run_idx = i; %#ok<AGROW>
    contrast_defs(end).direction = 'rating1_minus_rating4';
    contrast_defs(end).name = [prefix '_rating1_minus_rating4'];
    contrast_defs(end).weights = weights;
end

if isempty(contrast_defs)
    return;
end

matlabbatch = [];
matlabbatch{1}.spm.stats.con.spmmat = {spm_mat};
for c = 1:numel(contrast_defs)
    matlabbatch{1}.spm.stats.con.consess{c}.tcon.name = contrast_defs(c).name;
    matlabbatch{1}.spm.stats.con.consess{c}.tcon.weights = contrast_defs(c).weights;
    matlabbatch{1}.spm.stats.con.consess{c}.tcon.sessrep = 'none';
end
matlabbatch{1}.spm.stats.con.delete = 0;
spm_jobman('run', matlabbatch);

S = load(spm_mat, 'SPM');
if ~isfield(S.SPM, 'xCon')
    return;
end

model_dir = fileparts(spm_mat);
contrast_names = string({S.SPM.xCon.name});
for c = 1:numel(contrast_defs)
    idx = find(contrast_names == string(contrast_defs(c).name), 1, 'last');
    run_idx = contrast_defs(c).run_idx;
    direction = contrast_defs(c).direction;

    if isempty(idx) || ~isfield(S.SPM.xCon(idx), 'Vcon')
        contrast_lookup = set_contrast_lookup(contrast_lookup, run_idx, direction, '', false, ...
            'contrast_not_found_after_spm_con');
        continue;
    end

    con_file = fullfile(model_dir, S.SPM.xCon(idx).Vcon.fname);
    con_file = pick_existing_file(con_file);
    ok = ~isempty(con_file);
    if ok
        note = 'ok';
    else
        note = 'contrast_file_missing_after_spm_con';
    end
    contrast_lookup = set_contrast_lookup(contrast_lookup, run_idx, direction, con_file, ok, note);
end
end


function lookup = make_empty_contrast_lookup()
lookup = struct();
lookup.rating4_minus_rating1_file = '';
lookup.rating4_minus_rating1_estimated_successfully = false;
lookup.rating4_minus_rating1_note = '';
lookup.rating1_minus_rating4_file = '';
lookup.rating1_minus_rating4_estimated_successfully = false;
lookup.rating1_minus_rating4_note = '';
end


function contrast_lookup = set_contrast_lookup(contrast_lookup, run_idx, direction, con_file, ok, note)
if strcmp(direction, 'rating4_minus_rating1')
    contrast_lookup(run_idx).rating4_minus_rating1_file = con_file;
    contrast_lookup(run_idx).rating4_minus_rating1_estimated_successfully = ok;
    contrast_lookup(run_idx).rating4_minus_rating1_note = note;
else
    contrast_lookup(run_idx).rating1_minus_rating4_file = con_file;
    contrast_lookup(run_idx).rating1_minus_rating4_estimated_successfully = ok;
    contrast_lookup(run_idx).rating1_minus_rating4_note = note;
end
end


function cleanup_non_rating_betas(spm_mat, beta_lookup, do_estimate)
if ~do_estimate || exist(spm_mat, 'file') ~= 2
    return;
end

model_dir = fileparts(spm_mat);
keep_files = {};
for i = 1:numel(beta_lookup)
    if ~isempty(beta_lookup(i).rating1_beta_file)
        keep_files{end + 1} = beta_lookup(i).rating1_beta_file; %#ok<AGROW>
    end
    if ~isempty(beta_lookup(i).rating4_beta_file)
        keep_files{end + 1} = beta_lookup(i).rating4_beta_file; %#ok<AGROW>
    end
end
keep_files = unique(keep_files);

if isempty(keep_files)
    return;
end

d = dir(fullfile(model_dir, 'beta_*.nii'));
deleted_count = 0;
for i = 1:numel(d)
    beta_file = fullfile(d(i).folder, d(i).name);
    if ~any(strcmp(beta_file, keep_files))
        delete(beta_file);
        deleted_count = deleted_count + 1;
    end
end

if deleted_count > 0
    fprintf('Deleted %d non-rating beta files from %s\n', deleted_count, model_dir);
end
end


function rows = initialize_beta_rows(run_infos, spm_mat, model_dir, beta_lookup)
rows = struct([]);
for i = 1:numel(run_infos)
    rows = append_rows(rows, make_beta_row(run_infos(i), spm_mat, model_dir, ...
        'rating1_valuation', 1, run_infos(i).condition_info.n_rating1_trials, ...
        beta_lookup(i).rating1_beta_file, beta_lookup(i).rating1_beta_index, ...
        beta_lookup(i).rating1_estimated_successfully, beta_lookup(i).rating1_note));
    rows = append_rows(rows, make_beta_row(run_infos(i), spm_mat, model_dir, ...
        'rating4_valuation', 4, run_infos(i).condition_info.n_rating4_trials, ...
        beta_lookup(i).rating4_beta_file, beta_lookup(i).rating4_beta_index, ...
        beta_lookup(i).rating4_estimated_successfully, beta_lookup(i).rating4_note));
end
end


function row = make_beta_row(run_info, spm_mat, model_dir, condition, rating_value, n_trials, beta_file, beta_index, ok, note)
row = make_empty_beta_row();
row.sub = run_info.spec.sub;
row.session = run_info.spec.session_id;
row.run = run_info.spec.run;
row.spm_session_index = run_info.spm_session_index;
row.condition = condition;
row.rating_value = rating_value;
row.n_trials = n_trials;
row.condition_present = n_trials > 0;
row.has_pair_in_run = run_info.condition_info.n_rating1_trials > 0 && run_info.condition_info.n_rating4_trials > 0;
row.regressor_name = condition;
row.beta_index = beta_index;
row.beta_file = beta_file;
row.spm_mat = pick_existing_file(spm_mat);
row.model_dir = model_dir;
row.estimated_successfully = ok;
row.qc_run_spike_count = run_info.run_qc.spike_count;
row.qc_run_spike_fraction = run_info.run_qc.spike_fraction;
row.qc_run_nuisance_regressor_count = run_info.run_qc.n_regressors;
row.note = note;
end


function rows = initialize_contrast_rows(run_infos, spm_mat, contrast_lookup)
rows = struct([]);
for i = 1:numel(run_infos)
    rows = append_rows(rows, make_contrast_row(run_infos(i), spm_mat, ...
        'rating4_minus_rating1', contrast_lookup(i).rating4_minus_rating1_file, ...
        contrast_lookup(i).rating4_minus_rating1_estimated_successfully, ...
        contrast_lookup(i).rating4_minus_rating1_note));
    rows = append_rows(rows, make_contrast_row(run_infos(i), spm_mat, ...
        'rating1_minus_rating4', contrast_lookup(i).rating1_minus_rating4_file, ...
        contrast_lookup(i).rating1_minus_rating4_estimated_successfully, ...
        contrast_lookup(i).rating1_minus_rating4_note));
end
end


function row = make_contrast_row(run_info, spm_mat, contrast_name, contrast_file, ok, note)
row = make_empty_contrast_row();
row.sub = run_info.spec.sub;
row.session = run_info.spec.session_id;
row.run = run_info.spec.run;
row.spm_session_index = run_info.spm_session_index;
row.contrast_name = contrast_name;
row.n_rating1_trials = run_info.condition_info.n_rating1_trials;
row.n_rating4_trials = run_info.condition_info.n_rating4_trials;
row.has_pair_in_run = row.n_rating1_trials > 0 && row.n_rating4_trials > 0;
row.contrast_file = contrast_file;
row.spm_mat = pick_existing_file(spm_mat);
row.estimated_successfully = ok;
row.note = note;
end


function rows = initialize_summary_rows(run_infos, spm_mat, model_dir, beta_lookup, contrast_lookup)
rows = struct([]);
for i = 1:numel(run_infos)
    row = make_empty_summary_row();
    row.sub = run_infos(i).spec.sub;
    row.session = run_infos(i).spec.session_id;
    row.run = run_infos(i).spec.run;
    row.spm_session_index = run_infos(i).spm_session_index;
    row.n_target_trials = run_infos(i).condition_info.n_target_trials;
    row.n_rating1_trials = run_infos(i).condition_info.n_rating1_trials;
    row.n_rating4_trials = run_infos(i).condition_info.n_rating4_trials;
    row.n_other_valuation_trials = run_infos(i).condition_info.n_other_valuation_trials;
    row.has_rating1_beta = beta_lookup(i).rating1_estimated_successfully;
    row.has_rating4_beta = beta_lookup(i).rating4_estimated_successfully;
    row.has_pair_in_run = row.n_rating1_trials > 0 && row.n_rating4_trials > 0;
    row.has_pair_betas = row.has_rating1_beta && row.has_rating4_beta;
    row.has_run_contrasts = contrast_lookup(i).rating4_minus_rating1_estimated_successfully && ...
        contrast_lookup(i).rating1_minus_rating4_estimated_successfully;
    row.qc_run_spike_count = run_infos(i).run_qc.spike_count;
    row.qc_run_spike_fraction = run_infos(i).run_qc.spike_fraction;
    row.qc_run_nuisance_regressor_count = run_infos(i).run_qc.n_regressors;
    row.spm_mat = pick_existing_file(spm_mat);
    row.model_dir = model_dir;
    if row.has_pair_betas
        row.note = 'ok';
    else
        row.note = strjoin({beta_lookup(i).rating1_note, beta_lookup(i).rating4_note}, '; ');
    end
    rows = append_rows(rows, row);
end
end


function session_result = initialize_session_failure_result(session_specs, spm_mat, note)
session_result = make_empty_session_result();
model_dir = fileparts(spm_mat);
for i = 1:numel(session_specs)
    run_info = make_empty_run_info();
    run_info.spec = session_specs(i);
    run_info.spm_session_index = i;

    session_result.beta_rows = append_rows(session_result.beta_rows, make_beta_row( ...
        run_info, spm_mat, model_dir, 'rating1_valuation', 1, NaN, '', NaN, false, ...
        ['session_level_failure: ' note]));
    session_result.beta_rows = append_rows(session_result.beta_rows, make_beta_row( ...
        run_info, spm_mat, model_dir, 'rating4_valuation', 4, NaN, '', NaN, false, ...
        ['session_level_failure: ' note]));
    session_result.contrast_rows = append_rows(session_result.contrast_rows, make_contrast_row( ...
        run_info, spm_mat, 'rating4_minus_rating1', '', false, ['session_level_failure: ' note]));
    session_result.contrast_rows = append_rows(session_result.contrast_rows, make_contrast_row( ...
        run_info, spm_mat, 'rating1_minus_rating4', '', false, ['session_level_failure: ' note]));

    row = make_empty_summary_row();
    row.sub = session_specs(i).sub;
    row.session = session_specs(i).session_id;
    row.run = session_specs(i).run;
    row.spm_session_index = i;
    row.spm_mat = pick_existing_file(spm_mat);
    row.model_dir = model_dir;
    row.note = ['session_level_failure: ' note];
    session_result.summary_rows = append_rows(session_result.summary_rows, row);
end
end


function row = make_empty_beta_row()
row = struct();
row.sub = '';
row.session = '';
row.run = '';
row.spm_session_index = NaN;
row.condition = '';
row.rating_value = NaN;
row.n_trials = NaN;
row.condition_present = false;
row.has_pair_in_run = false;
row.regressor_name = '';
row.beta_index = NaN;
row.beta_file = '';
row.spm_mat = '';
row.model_dir = '';
row.estimated_successfully = false;
row.qc_run_spike_count = NaN;
row.qc_run_spike_fraction = NaN;
row.qc_run_nuisance_regressor_count = NaN;
row.note = '';
end


function row = make_empty_contrast_row()
row = struct();
row.sub = '';
row.session = '';
row.run = '';
row.spm_session_index = NaN;
row.contrast_name = '';
row.n_rating1_trials = NaN;
row.n_rating4_trials = NaN;
row.has_pair_in_run = false;
row.contrast_file = '';
row.spm_mat = '';
row.estimated_successfully = false;
row.note = '';
end


function row = make_empty_summary_row()
row = struct();
row.sub = '';
row.session = '';
row.run = '';
row.spm_session_index = NaN;
row.n_target_trials = NaN;
row.n_rating1_trials = NaN;
row.n_rating4_trials = NaN;
row.n_other_valuation_trials = NaN;
row.has_rating1_beta = false;
row.has_rating4_beta = false;
row.has_pair_in_run = false;
row.has_pair_betas = false;
row.has_run_contrasts = false;
row.qc_run_spike_count = NaN;
row.qc_run_spike_fraction = NaN;
row.qc_run_nuisance_regressor_count = NaN;
row.spm_mat = '';
row.model_dir = '';
row.note = '';
end


function results = force_nonempty_results(results, empty_row)
if isempty(results)
    results = empty_row;
    results.note = 'no_rows_generated';
end
end


function results = concat_result_field(results_cells, field_name)
results = struct([]);
for i = 1:numel(results_cells)
    if isempty(results_cells{i}) || ~isfield(results_cells{i}, field_name)
        continue;
    end
    results = append_rows(results, results_cells{i}.(field_name));
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
[~, name] = fileparts(source_file);
[~, base_name] = fileparts(name);
nii_file = fullfile(cache_dir, [base_name '.nii']);
if exist(nii_file, 'file') == 2
    return;
end
gunzip(source_file, cache_dir);
[~, name] = fileparts(source_file);
[~, base_name] = fileparts(name);
nii_file = fullfile(cache_dir, [base_name '.nii']);
must_exist(nii_file, 'Cached NIfTI file');
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
