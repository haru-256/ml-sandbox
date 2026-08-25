/**
 * Find target projects from changed files, or all packages when run-all files change.
 *
 * Inputs are passed via environment variables so this module can be loaded by
 * actions/github-script with `require`:
 * - TARGET_PROJECTS: comma-separated project paths. If set, used as-is.
 * - REQUIRED_FILES: comma-separated file names that identify a project.
 * - MAX_DEPTH: maximum directory depth of a project relative to the workspace.
 * - CHANGED_FILES: JSON array of changed file/directory paths.
 * - RUN_ALL: "true" when a caller-supplied glob list has changes.
 *
 * @param {object} params
 * @param {import('@actions/core')} params.core
 */
module.exports = async ({ core }) => {
  const { execFileSync } = require('node:child_process');
  const fs = require('node:fs');

  const workspace = process.env.GITHUB_WORKSPACE || process.cwd();
  const dispatchInput = (process.env.TARGET_PROJECTS ?? '').trim();
  if (dispatchInput !== '') {
    const projects = dispatchInput
      .split(',')
      .map((path) => path.trim())
      .filter(Boolean);
    core.setOutput('projects', JSON.stringify(projects));
    return;
  }

  const requiredFiles = (process.env.REQUIRED_FILES ?? '')
    .split(',')
    .map((path) => path.trim())
    .filter(Boolean);
  if (requiredFiles.length === 0) {
    core.setFailed('REQUIRED_FILES must not be empty');
    return;
  }

  const maxDepth = parseInt(process.env.MAX_DEPTH ?? '', 10);
  if (!Number.isInteger(maxDepth) || maxDepth < 1) {
    core.setFailed('MAX_DEPTH must be a positive integer');
    return;
  }

  function isProject(relPath) {
    return requiredFiles.every((file) => fs.existsSync(`${workspace}/${relPath}/${file}`));
  }

  function ancestors(changedPaths) {
    const set = new Set();
    changedPaths.forEach((changedPath) => {
      const segments = changedPath.split('/');
      let current = '';
      segments.forEach((segment, index) => {
        current = index === 0 ? segment : `${current}/${segment}`;
        set.add(current);
      });
    });
    return Array.from(set);
  }

  function dirsUpTo(depth) {
    const marker = requiredFiles[0];
    const out = execFileSync('git', ['ls-files', '-z', '--', `:(glob)**/${marker}`], {
      cwd: workspace,
      encoding: 'utf8',
    });
    const dirs = new Set();
    for (const file of out.split('\0').filter(Boolean)) {
      const lastSlash = file.lastIndexOf('/');
      if (lastSlash <= 0) {
        continue;
      }
      const dir = file.slice(0, lastSlash);
      if (dir.split('/').length <= depth) {
        dirs.add(dir);
      }
    }
    return Array.from(dirs);
  }

  const runAll = process.env.RUN_ALL === 'true';
  const changedFiles = JSON.parse((process.env.CHANGED_FILES ?? '').trim() || '[]');
  const candidates = runAll ? dirsUpTo(maxDepth) : ancestors(changedFiles);
  const projects = candidates.filter(isProject).sort();
  core.setOutput('projects', JSON.stringify(projects));
};
