use crate::client::lifecycle::{MAX_REPORTED_PROCESSES, process_stat};

pub(crate) fn cleanup_diagnostics(pid: u32) -> String {
    #[cfg(target_os = "linux")]
    {
        let mut report = Vec::new();
        match process_stat(pid) {
            Ok(Some(stat)) => report.push(format!(
                "direct_child={pid} state={} ppid={} pgid={}",
                stat.state, stat.ppid, stat.pgrp
            )),
            Ok(None) => report.push(format!("direct_child={pid} absent")),
            Err(error) => report.push(format!("direct_child={pid} stat_error={error}")),
        }
        report.push(group_members_diagnostic(pid));
        report.push(threads_diagnostic(pid));
        report.join(" ")
    }
    #[cfg(not(target_os = "linux"))]
    {
        format!("direct_child={pid} state=unknown")
    }
}

#[cfg(target_os = "linux")]
fn group_members_diagnostic(pid: u32) -> String {
    let group = i32::try_from(pid).ok();
    let mut members = Vec::new();
    let mut count = 0usize;
    match std::fs::read_dir("/proc") {
        Ok(entries) => {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let Ok(member) = name.to_string_lossy().parse::<u32>() else {
                    continue;
                };
                let Ok(Some(stat)) = process_stat(member) else {
                    continue;
                };
                if Some(stat.pgrp) == group && !matches!(stat.state, 'Z' | 'X') {
                    count += 1;
                    if members.len() < MAX_REPORTED_PROCESSES {
                        members.push(format!("{member}:{}", stat.state));
                    }
                }
            }
        }
        Err(error) => return format!("group_scan_error={error}"),
    }
    let suffix = if count > members.len() {
        format!(" +{} more", count - members.len())
    } else {
        String::new()
    };
    format!("group_live={count} [{}]{suffix}", members.join(","))
}

#[cfg(target_os = "linux")]
fn threads_diagnostic(pid: u32) -> String {
    let Ok(entries) = std::fs::read_dir(format!("/proc/{pid}/task")) else {
        return "threads=absent".to_owned();
    };
    let mut states = std::collections::BTreeMap::new();
    let mut blocked = Vec::new();
    let mut count = 0usize;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Ok(tid) = name.to_string_lossy().parse::<u32>() else {
            continue;
        };
        let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/stat")) else {
            continue;
        };
        let Some((_, fields)) = stat.rsplit_once(") ") else {
            continue;
        };
        let Some(state) = fields
            .split_whitespace()
            .next()
            .and_then(|value| value.chars().next())
        else {
            continue;
        };
        count += 1;
        *states.entry(state).or_insert(0usize) += 1;
        if !matches!(state, 'R' | 'S' | 'I') && blocked.len() < MAX_REPORTED_PROCESSES {
            let wait_channel = std::fs::read_to_string(format!("/proc/{pid}/task/{tid}/wchan"))
                .map(|value| value.trim().to_owned())
                .unwrap_or_default();
            blocked.push(if wait_channel.is_empty() {
                format!("{tid}:{state}")
            } else {
                format!("{tid}:{state}:{wait_channel}")
            });
        }
    }
    let summary = states
        .into_iter()
        .map(|(state, count)| format!("{state}={count}"))
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "threads={count} [{summary}] blocked=[{}]",
        blocked.join(",")
    )
}
