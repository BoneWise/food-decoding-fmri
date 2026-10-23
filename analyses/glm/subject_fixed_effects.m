% Subject-level fixed-effects approximation from run-level contrast images.
% For each subject and each contrast (1>4, 4>1), fit a one-sample second-level
% model across the subject's run-level contrasts and write a subject-level
% contrast image for group analysis.

clearvars -except spm_dir output_root contrast_index_csv overwrite_existing
clc;

if ~exist('spm_dir', 'var') || isempty(spm_dir)
    spm_dir = getenv('SPM_DIR');
end
if ~exist('output_root', 'var') || isempty(output_root)
    output_root = getenv('FOODPREF_GLM_S6_ROOT');
end
if ~exist('contrast_index_csv', 'var') || isempty(contrast_index_csv)
    contrast_index_csv = fullfile(output_root, 'contrast_index.csv');
end
if ~exist('overwrite_existing', 'var') || isempty(overwrite_existing)
    overwrite_existing = true;
end

if ~isempty(spm_dir), addpath(spm_dir, '-begin'); end
spm('Defaults', 'fMRI');
spm_get_defaults('cmdline', true);
spm_jobman('initcfg');

index_table = readtable(contrast_index_csv, 'FileType', 'text');
required_columns = {
    'sub', 'session_id', 'run', ...
    'con_1gt4', 'con_4gt1', ...
    'contrast_rating1_gt_rating4_available', ...
    'contrast_rating4_gt_rating1_available'
};
must_have_columns(index_table, required_columns);

fixed_root = ensure_dir(fullfile(output_root, 'second_level_fixed_effects'));

contrast_specs = {
    struct('name', '1gt4', 'column', 'con_1gt4', 'available_column', 'contrast_rating1_gt_rating4_available'), ...
    struct('name', '4gt1', 'column', 'con_4gt1', 'available_column', 'contrast_rating4_gt_rating1_available')
};

subjects = unique(string(index_table.sub), 'stable');
summary_rows = struct([]);

for subj_idx = 1:numel(subjects)
    subject_id = char(subjects(subj_idx));
    subject_mask = string(index_table.sub) == string(subject_id);
    subject_rows = index_table(subject_mask, :);

    for contrast_idx = 1:numel(contrast_specs)
        spec = contrast_specs{contrast_idx};
        available_mask = subject_rows.(spec.available_column) == 1;
        contrast_paths = cellstr(string(subject_rows.(spec.column)(available_mask)));
        contrast_paths = contrast_paths(~cellfun(@isempty, contrast_paths));

        row = make_empty_summary_row();
        row.sub = subject_id;
        row.contrast_name = spec.name;
        row.n_run_level_images = numel(contrast_paths);

        subject_dir = fullfile(fixed_root, spec.name, subject_id);
        row.output_dir = subject_dir;

        if isempty(contrast_paths)
            row.status = 'skipped_no_available_run_level_contrasts';
            summary_rows = append_rows(summary_rows, row);
            continue;
        end

        if exist(subject_dir, 'dir') == 7 && overwrite_existing
            rmdir(subject_dir, 's');
        end
        ensure_dir(subject_dir);

        spm_mat = fullfile(subject_dir, 'SPM.mat');

        try
            scans = reshape(cellfun(@char, contrast_paths, 'UniformOutput', false), [], 1);

            matlabbatch = [];
            matlabbatch{1}.spm.stats.factorial_design.dir = {subject_dir};
            matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = scans;
            matlabbatch{1}.spm.stats.factorial_design.cov = struct('c', {}, 'cname', {}, 'iCFI', {}, 'iCC', {});
            matlabbatch{1}.spm.stats.factorial_design.multi_cov = struct('files', {}, 'iCFI', {}, 'iCC', {});
            matlabbatch{1}.spm.stats.factorial_design.masking.tm.tm_none = 1;
            matlabbatch{1}.spm.stats.factorial_design.masking.im = 1;
            matlabbatch{1}.spm.stats.factorial_design.masking.em = {''};
            matlabbatch{1}.spm.stats.factorial_design.globalc.g_omit = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.gmsca.gmsca_no = 1;
            matlabbatch{1}.spm.stats.factorial_design.globalm.glonorm = 1;

            matlabbatch{2}.spm.stats.fmri_est.spmmat = {spm_mat};
            matlabbatch{2}.spm.stats.fmri_est.write_residuals = 0;
            matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;

            matlabbatch{3}.spm.stats.con.spmmat = {spm_mat};
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'subject_mean';
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
            matlabbatch{3}.spm.stats.con.delete = 1;

            spm_jobman('run', matlabbatch);

            row.spm_mat = pick_existing_file(spm_mat);
            row.mask_path = pick_existing_file(fullfile(subject_dir, 'mask.nii'));
            row.subject_contrast_path = pick_existing_file(fullfile(subject_dir, 'con_0001.nii'));
            row.subject_spmT_path = pick_existing_file(fullfile(subject_dir, 'spmT_0001.nii'));
            row.status = ternary(~isempty(row.subject_contrast_path), 'ok', 'missing_subject_contrast_output');
        catch ME
            row.status = 'error';
            row.note = simplify_error_message(ME);
        end

        summary_rows = append_rows(summary_rows, row);
    end
end

summary_table = struct2table(force_nonempty_results(summary_rows), 'AsArray', true);
summary_csv = fullfile(output_root, 'second_level_fixed_effects_summary.csv');
writetable(summary_table, summary_csv);

fprintf('\nSecond-level fixed-effects summary: %s\n', summary_csv);
fprintf('Rows written: %d\n', height(summary_table));
fprintf('Successful subject-level outputs: %d\n', nnz(strcmp(summary_table.status, 'ok')));
fprintf('Skipped because no available run-level contrasts: %d\n', nnz(strcmp(summary_table.status, 'skipped_no_available_run_level_contrasts')));
fprintf('Errors: %d\n', nnz(strcmp(summary_table.status, 'error')));


function pathstr = pick_existing_file(pathstr)
if isempty(pathstr) || exist(pathstr, 'file') ~= 2
    pathstr = '';
end
end


function pathstr = ensure_dir(pathstr)
if exist(pathstr, 'dir') ~= 7
    mkdir(pathstr);
end
end


function must_have_columns(tbl, names)
missing = names(~ismember(names, tbl.Properties.VariableNames));
if ~isempty(missing)
    error('Missing required column(s): %s', strjoin(missing, ', '));
end
end


function row = make_empty_summary_row()
row = struct();
row.sub = '';
row.contrast_name = '';
row.n_run_level_images = 0;
row.output_dir = '';
row.spm_mat = '';
row.mask_path = '';
row.subject_contrast_path = '';
row.subject_spmT_path = '';
row.status = '';
row.note = '';
end


function rows = append_rows(rows, new_rows)
if isempty(new_rows)
    return;
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


function results = force_nonempty_results(results)
if isempty(results)
    results = make_empty_summary_row();
    results.status = 'no_rows';
end
end


function out = ternary(condition, when_true, when_false)
if condition
    out = when_true;
else
    out = when_false;
end
end


function text = simplify_error_message(ME)
text = regexprep(ME.message, '\s+', ' ');
end
