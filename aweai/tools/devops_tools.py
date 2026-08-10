from __future__ import annotations

from aweai.tools.registry import tool
from typing import Any, Dict, List, Optional
from subprocess import run, PIPE
from shutil import which


def _bool_to_str(v: Any) -> str:
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


def _execute(binary: str, subcmd: List[str], **kwargs: Any) -> Dict[str, Any]:
    if which(binary) is None:
        return {"error": f"{binary} not found in PATH"}
    cmd = [binary] + subcmd
    for key, val in kwargs.items():
        if val is None:
            continue
        if key.startswith("-"):
            cmd.append(key)
            cmd.append(_bool_to_str(val))
        elif key.startswith("--"):
            cmd.append(key + "=" + _bool_to_str(val))
        else:
            cmd.append(key + "=" + _bool_to_str(val))
    try:
        result = run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {"error": result.stderr.strip() or result.stdout.strip()}
        return {"stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as e:
        return {"error": str(e)}

@tool("docker_build", "devops", "Execute docker_build command")
def docker_build(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['build'], **kwargs)

@tool("docker_run", "devops", "Execute docker_run command")
def docker_run(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['run'], **kwargs)

@tool("docker_stop", "devops", "Execute docker_stop command")
def docker_stop(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['stop'], **kwargs)

@tool("docker_remove", "devops", "Execute docker_remove command")
def docker_remove(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['remove'], **kwargs)

@tool("docker_ps", "devops", "Execute docker_ps command")
def docker_ps(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['ps'], **kwargs)

@tool("docker_images", "devops", "Execute docker_images command")
def docker_images(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['images'], **kwargs)

@tool("docker_volumes", "devops", "Execute docker_volumes command")
def docker_volumes(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volumes'], **kwargs)

@tool("docker_networks", "devops", "Execute docker_networks command")
def docker_networks(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['networks'], **kwargs)

@tool("docker_logs", "devops", "Execute docker_logs command")
def docker_logs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['logs'], **kwargs)

@tool("docker_exec", "devops", "Execute docker_exec command")
def docker_exec(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['exec'], **kwargs)

@tool("docker_copy", "devops", "Execute docker_copy command")
def docker_copy(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['copy'], **kwargs)

@tool("docker_inspect", "devops", "Execute docker_inspect command")
def docker_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['inspect'], **kwargs)

@tool("docker_tag", "devops", "Execute docker_tag command")
def docker_tag(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['tag'], **kwargs)

@tool("docker_push", "devops", "Execute docker_push command")
def docker_push(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['push'], **kwargs)

@tool("docker_pull", "devops", "Execute docker_pull command")
def docker_pull(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['pull'], **kwargs)

@tool("docker_login", "devops", "Execute docker_login command")
def docker_login(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['login'], **kwargs)

@tool("docker_logout", "devops", "Execute docker_logout command")
def docker_logout(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['logout'], **kwargs)

@tool("docker_search", "devops", "Execute docker_search command")
def docker_search(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['search'], **kwargs)

@tool("docker_save", "devops", "Execute docker_save command")
def docker_save(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['save'], **kwargs)

@tool("docker_load", "devops", "Execute docker_load command")
def docker_load(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['load'], **kwargs)

@tool("docker_import", "devops", "Execute docker_import command")
def docker_import(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['import'], **kwargs)

@tool("docker_export", "devops", "Execute docker_export command")
def docker_export(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['export'], **kwargs)

@tool("docker_commit", "devops", "Execute docker_commit command")
def docker_commit(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['commit'], **kwargs)

@tool("docker_diff", "devops", "Execute docker_diff command")
def docker_diff(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['diff'], **kwargs)

@tool("docker_events", "devops", "Execute docker_events command")
def docker_events(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['events'], **kwargs)

@tool("docker_history", "devops", "Execute docker_history command")
def docker_history(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['history'], **kwargs)

@tool("docker_info", "devops", "Execute docker_info command")
def docker_info(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['info'], **kwargs)

@tool("docker_version", "devops", "Execute docker_version command")
def docker_version(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['version'], **kwargs)

@tool("docker_swarm_init", "devops", "Execute docker_swarm_init command")
def docker_swarm_init(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'init'], **kwargs)

@tool("docker_swarm_join", "devops", "Execute docker_swarm_join command")
def docker_swarm_join(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'join'], **kwargs)

@tool("docker_swarm_leave", "devops", "Execute docker_swarm_leave command")
def docker_swarm_leave(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'leave'], **kwargs)

@tool("docker_swarm_update", "devops", "Execute docker_swarm_update command")
def docker_swarm_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'update'], **kwargs)

@tool("docker_swarm_inspect", "devops", "Execute docker_swarm_inspect command")
def docker_swarm_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'inspect'], **kwargs)

@tool("docker_swarm_unlock", "devops", "Execute docker_swarm_unlock command")
def docker_swarm_unlock(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'unlock'], **kwargs)

@tool("docker_swarm_unlockkey", "devops", "Execute docker_swarm_unlockkey command")
def docker_swarm_unlockkey(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['swarm', 'unlockkey'], **kwargs)

@tool("docker_stack_deploy", "devops", "Execute docker_stack_deploy command")
def docker_stack_deploy(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['stack', 'deploy'], **kwargs)

@tool("docker_stack_rm", "devops", "Execute docker_stack_rm command")
def docker_stack_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['stack', 'rm'], **kwargs)

@tool("docker_stack_ps", "devops", "Execute docker_stack_ps command")
def docker_stack_ps(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['stack', 'ps'], **kwargs)

@tool("docker_stack_services", "devops", "Execute docker_stack_services command")
def docker_stack_services(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['stack', 'services'], **kwargs)

@tool("docker_service_create", "devops", "Execute docker_service_create command")
def docker_service_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'create'], **kwargs)

@tool("docker_service_inspect", "devops", "Execute docker_service_inspect command")
def docker_service_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'inspect'], **kwargs)

@tool("docker_service_logs", "devops", "Execute docker_service_logs command")
def docker_service_logs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'logs'], **kwargs)

@tool("docker_service_ps", "devops", "Execute docker_service_ps command")
def docker_service_ps(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'ps'], **kwargs)

@tool("docker_service_rm", "devops", "Execute docker_service_rm command")
def docker_service_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'rm'], **kwargs)

@tool("docker_service_scale", "devops", "Execute docker_service_scale command")
def docker_service_scale(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'scale'], **kwargs)

@tool("docker_service_update", "devops", "Execute docker_service_update command")
def docker_service_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['service', 'update'], **kwargs)

@tool("docker_config_create", "devops", "Execute docker_config_create command")
def docker_config_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['config', 'create'], **kwargs)

@tool("docker_config_inspect", "devops", "Execute docker_config_inspect command")
def docker_config_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['config', 'inspect'], **kwargs)

@tool("docker_config_ls", "devops", "Execute docker_config_ls command")
def docker_config_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['config', 'ls'], **kwargs)

@tool("docker_config_rm", "devops", "Execute docker_config_rm command")
def docker_config_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['config', 'rm'], **kwargs)

@tool("docker_secret_create", "devops", "Execute docker_secret_create command")
def docker_secret_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['secret', 'create'], **kwargs)

@tool("docker_secret_inspect", "devops", "Execute docker_secret_inspect command")
def docker_secret_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['secret', 'inspect'], **kwargs)

@tool("docker_secret_ls", "devops", "Execute docker_secret_ls command")
def docker_secret_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['secret', 'ls'], **kwargs)

@tool("docker_secret_rm", "devops", "Execute docker_secret_rm command")
def docker_secret_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['secret', 'rm'], **kwargs)

@tool("docker_node_inspect", "devops", "Execute docker_node_inspect command")
def docker_node_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'inspect'], **kwargs)

@tool("docker_node_ls", "devops", "Execute docker_node_ls command")
def docker_node_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'ls'], **kwargs)

@tool("docker_node_promote", "devops", "Execute docker_node_promote command")
def docker_node_promote(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'promote'], **kwargs)

@tool("docker_node_demote", "devops", "Execute docker_node_demote command")
def docker_node_demote(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'demote'], **kwargs)

@tool("docker_node_update", "devops", "Execute docker_node_update command")
def docker_node_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'update'], **kwargs)

@tool("docker_node_rm", "devops", "Execute docker_node_rm command")
def docker_node_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['node', 'rm'], **kwargs)

@tool("docker_task_inspect", "devops", "Execute docker_task_inspect command")
def docker_task_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['task', 'inspect'], **kwargs)

@tool("docker_task_ls", "devops", "Execute docker_task_ls command")
def docker_task_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['task', 'ls'], **kwargs)

@tool("docker_task_logs", "devops", "Execute docker_task_logs command")
def docker_task_logs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['task', 'logs'], **kwargs)

@tool("docker_task_rm", "devops", "Execute docker_task_rm command")
def docker_task_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['task', 'rm'], **kwargs)

@tool("docker_plugin_inspect", "devops", "Execute docker_plugin_inspect command")
def docker_plugin_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['plugin', 'inspect'], **kwargs)

@tool("docker_plugin_ls", "devops", "Execute docker_plugin_ls command")
def docker_plugin_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['plugin', 'ls'], **kwargs)

@tool("docker_plugin_install", "devops", "Execute docker_plugin_install command")
def docker_plugin_install(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['plugin', 'install'], **kwargs)

@tool("docker_plugin_rm", "devops", "Execute docker_plugin_rm command")
def docker_plugin_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['plugin', 'rm'], **kwargs)

@tool("docker_plugin_upgrade", "devops", "Execute docker_plugin_upgrade command")
def docker_plugin_upgrade(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['plugin', 'upgrade'], **kwargs)

@tool("docker_volume_create", "devops", "Execute docker_volume_create command")
def docker_volume_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volume', 'create'], **kwargs)

@tool("docker_volume_inspect", "devops", "Execute docker_volume_inspect command")
def docker_volume_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volume', 'inspect'], **kwargs)

@tool("docker_volume_ls", "devops", "Execute docker_volume_ls command")
def docker_volume_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volume', 'ls'], **kwargs)

@tool("docker_volume_rm", "devops", "Execute docker_volume_rm command")
def docker_volume_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volume', 'rm'], **kwargs)

@tool("docker_volume_prune", "devops", "Execute docker_volume_prune command")
def docker_volume_prune(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['volume', 'prune'], **kwargs)

@tool("docker_network_create", "devops", "Execute docker_network_create command")
def docker_network_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'create'], **kwargs)

@tool("docker_network_inspect", "devops", "Execute docker_network_inspect command")
def docker_network_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'inspect'], **kwargs)

@tool("docker_network_ls", "devops", "Execute docker_network_ls command")
def docker_network_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'ls'], **kwargs)

@tool("docker_network_rm", "devops", "Execute docker_network_rm command")
def docker_network_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'rm'], **kwargs)

@tool("docker_network_connect", "devops", "Execute docker_network_connect command")
def docker_network_connect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'connect'], **kwargs)

@tool("docker_network_disconnect", "devops", "Execute docker_network_disconnect command")
def docker_network_disconnect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'disconnect'], **kwargs)

@tool("docker_network_prune", "devops", "Execute docker_network_prune command")
def docker_network_prune(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['network', 'prune'], **kwargs)

@tool("docker_buildx_build", "devops", "Execute docker_buildx_build command")
def docker_buildx_build(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'build'], **kwargs)

@tool("docker_buildx_create", "devops", "Execute docker_buildx_create command")
def docker_buildx_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'create'], **kwargs)

@tool("docker_buildx_inspect", "devops", "Execute docker_buildx_inspect command")
def docker_buildx_inspect(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'inspect'], **kwargs)

@tool("docker_buildx_ls", "devops", "Execute docker_buildx_ls command")
def docker_buildx_ls(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'ls'], **kwargs)

@tool("docker_buildx_rm", "devops", "Execute docker_buildx_rm command")
def docker_buildx_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'rm'], **kwargs)

@tool("docker_buildx_stop", "devops", "Execute docker_buildx_stop command")
def docker_buildx_stop(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'stop'], **kwargs)

@tool("docker_buildx_start", "devops", "Execute docker_buildx_start command")
def docker_buildx_start(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'start'], **kwargs)

@tool("docker_buildx_upgrade", "devops", "Execute docker_buildx_upgrade command")
def docker_buildx_upgrade(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'upgrade'], **kwargs)

@tool("docker_buildx_use", "devops", "Execute docker_buildx_use command")
def docker_buildx_use(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['buildx', 'use'], **kwargs)

@tool("docker_compose_up", "devops", "Execute docker_compose_up command")
def docker_compose_up(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'up'], **kwargs)

@tool("docker_compose_down", "devops", "Execute docker_compose_down command")
def docker_compose_down(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'down'], **kwargs)

@tool("docker_compose_start", "devops", "Execute docker_compose_start command")
def docker_compose_start(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'start'], **kwargs)

@tool("docker_compose_stop", "devops", "Execute docker_compose_stop command")
def docker_compose_stop(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'stop'], **kwargs)

@tool("docker_compose_restart", "devops", "Execute docker_compose_restart command")
def docker_compose_restart(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'restart'], **kwargs)

@tool("docker_compose_ps", "devops", "Execute docker_compose_ps command")
def docker_compose_ps(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'ps'], **kwargs)

@tool("docker_compose_logs", "devops", "Execute docker_compose_logs command")
def docker_compose_logs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'logs'], **kwargs)

@tool("docker_compose_build", "devops", "Execute docker_compose_build command")
def docker_compose_build(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'build'], **kwargs)

@tool("docker_compose_pull", "devops", "Execute docker_compose_pull command")
def docker_compose_pull(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'pull'], **kwargs)

@tool("docker_compose_push", "devops", "Execute docker_compose_push command")
def docker_compose_push(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'push'], **kwargs)

@tool("docker_compose_config", "devops", "Execute docker_compose_config command")
def docker_compose_config(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'config'], **kwargs)

@tool("docker_compose_create", "devops", "Execute docker_compose_create command")
def docker_compose_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'create'], **kwargs)

@tool("docker_compose_exec", "devops", "Execute docker_compose_exec command")
def docker_compose_exec(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'exec'], **kwargs)

@tool("docker_compose_run", "devops", "Execute docker_compose_run command")
def docker_compose_run(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'run'], **kwargs)

@tool("docker_compose_rm", "devops", "Execute docker_compose_rm command")
def docker_compose_rm(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'rm'], **kwargs)

@tool("docker_compose_scale", "devops", "Execute docker_compose_scale command")
def docker_compose_scale(**kwargs: Any) -> Dict[str, Any]:
    return _execute("docker", ['compose', 'scale'], **kwargs)

@tool("k8s_apply", "devops", "Execute k8s_apply command")
def k8s_apply(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['apply'], **kwargs)

@tool("k8s_delete", "devops", "Execute k8s_delete command")
def k8s_delete(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['delete'], **kwargs)

@tool("k8s_get", "devops", "Execute k8s_get command")
def k8s_get(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['get'], **kwargs)

@tool("k8s_describe", "devops", "Execute k8s_describe command")
def k8s_describe(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['describe'], **kwargs)

@tool("k8s_logs", "devops", "Execute k8s_logs command")
def k8s_logs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['logs'], **kwargs)

@tool("k8s_exec", "devops", "Execute k8s_exec command")
def k8s_exec(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['exec'], **kwargs)

@tool("k8s_port_forward", "devops", "Execute k8s_port_forward command")
def k8s_port_forward(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['port_forward'], **kwargs)

@tool("k8s_proxy", "devops", "Execute k8s_proxy command")
def k8s_proxy(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['proxy'], **kwargs)

@tool("k8s_top", "devops", "Execute k8s_top command")
def k8s_top(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['top'], **kwargs)

@tool("k8s_apiversions", "devops", "Execute k8s_apiversions command")
def k8s_apiversions(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['apiversions'], **kwargs)

@tool("k8s_resources", "devops", "Execute k8s_resources command")
def k8s_resources(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['resources'], **kwargs)

@tool("k8s_namespaces", "devops", "Execute k8s_namespaces command")
def k8s_namespaces(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['namespaces'], **kwargs)

@tool("k8s_nodes", "devops", "Execute k8s_nodes command")
def k8s_nodes(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['nodes'], **kwargs)

@tool("k8s_pods", "devops", "Execute k8s_pods command")
def k8s_pods(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['pods'], **kwargs)

@tool("k8s_services", "devops", "Execute k8s_services command")
def k8s_services(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['services'], **kwargs)

@tool("k8s_deployments", "devops", "Execute k8s_deployments command")
def k8s_deployments(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['deployments'], **kwargs)

@tool("k8s_replicasets", "devops", "Execute k8s_replicasets command")
def k8s_replicasets(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['replicasets'], **kwargs)

@tool("k8s_statefulsets", "devops", "Execute k8s_statefulsets command")
def k8s_statefulsets(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['statefulsets'], **kwargs)

@tool("k8s_daemonsets", "devops", "Execute k8s_daemonsets command")
def k8s_daemonsets(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['daemonsets'], **kwargs)

@tool("k8s_jobs", "devops", "Execute k8s_jobs command")
def k8s_jobs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['jobs'], **kwargs)

@tool("k8s_cronjobs", "devops", "Execute k8s_cronjobs command")
def k8s_cronjobs(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['cronjobs'], **kwargs)

@tool("k8s_configmaps", "devops", "Execute k8s_configmaps command")
def k8s_configmaps(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['configmaps'], **kwargs)

@tool("k8s_secrets", "devops", "Execute k8s_secrets command")
def k8s_secrets(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['secrets'], **kwargs)

@tool("k8s_serviceaccounts", "devops", "Execute k8s_serviceaccounts command")
def k8s_serviceaccounts(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['serviceaccounts'], **kwargs)

@tool("k8s_roles", "devops", "Execute k8s_roles command")
def k8s_roles(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['roles'], **kwargs)

@tool("k8s_rolebindings", "devops", "Execute k8s_rolebindings command")
def k8s_rolebindings(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['rolebindings'], **kwargs)

@tool("k8s_clusterroles", "devops", "Execute k8s_clusterroles command")
def k8s_clusterroles(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['clusterroles'], **kwargs)

@tool("k8s_clusterrolebindings", "devops", "Execute k8s_clusterrolebindings command")
def k8s_clusterrolebindings(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['clusterrolebindings'], **kwargs)

@tool("k8s_networkpolicies", "devops", "Execute k8s_networkpolicies command")
def k8s_networkpolicies(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['networkpolicies'], **kwargs)

@tool("k8s_storageclasses", "devops", "Execute k8s_storageclasses command")
def k8s_storageclasses(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['storageclasses'], **kwargs)

@tool("k8s_persistentvolumes", "devops", "Execute k8s_persistentvolumes command")
def k8s_persistentvolumes(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['persistentvolumes'], **kwargs)

@tool("k8s_persistentvolumeclaims", "devops", "Execute k8s_persistentvolumeclaims command")
def k8s_persistentvolumeclaims(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['persistentvolumeclaims'], **kwargs)

@tool("k8s_ingresses", "devops", "Execute k8s_ingresses command")
def k8s_ingresses(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['ingresses'], **kwargs)

@tool("k8s_ingressclasses", "devops", "Execute k8s_ingressclasses command")
def k8s_ingressclasses(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['ingressclasses'], **kwargs)

@tool("k8s_certificatesigningrequests", "devops", "Execute k8s_certificatesigningrequests command")
def k8s_certificatesigningrequests(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['certificatesigningrequests'], **kwargs)

@tool("k8s_leases", "devops", "Execute k8s_leases command")
def k8s_leases(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['leases'], **kwargs)

@tool("k8s_events", "devops", "Execute k8s_events command")
def k8s_events(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['events'], **kwargs)

@tool("k8s_endpoints", "devops", "Execute k8s_endpoints command")
def k8s_endpoints(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['endpoints'], **kwargs)

@tool("k8s_endpointslices", "devops", "Execute k8s_endpointslices command")
def k8s_endpointslices(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['endpointslices'], **kwargs)

@tool("k8s_hpas", "devops", "Execute k8s_hpas command")
def k8s_hpas(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['hpas'], **kwargs)

@tool("k8s_poddisruptionbudgets", "devops", "Execute k8s_poddisruptionbudgets command")
def k8s_poddisruptionbudgets(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['poddisruptionbudgets'], **kwargs)

@tool("k8s_priorityclasses", "devops", "Execute k8s_priorityclasses command")
def k8s_priorityclasses(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['priorityclasses'], **kwargs)

@tool("k8s_runtimeclasses", "devops", "Execute k8s_runtimeclasses command")
def k8s_runtimeclasses(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['runtimeclasses'], **kwargs)

@tool("k8s_resourcequotas", "devops", "Execute k8s_resourcequotas command")
def k8s_resourcequotas(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['resourcequotas'], **kwargs)

@tool("k8s_limitranges", "devops", "Execute k8s_limitranges command")
def k8s_limitranges(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['limitranges'], **kwargs)

@tool("k8s_mutatingwebhookconfigurations", "devops", "Execute k8s_mutatingwebhookconfigurations command")
def k8s_mutatingwebhookconfigurations(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['mutatingwebhookconfigurations'], **kwargs)

@tool("k8s_validatingwebhookconfigurations", "devops", "Execute k8s_validatingwebhookconfigurations command")
def k8s_validatingwebhookconfigurations(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['validatingwebhookconfigurations'], **kwargs)

@tool("k8s_csidrivers", "devops", "Execute k8s_csidrivers command")
def k8s_csidrivers(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['csidrivers'], **kwargs)

@tool("k8s_csinodes", "devops", "Execute k8s_csinodes command")
def k8s_csinodes(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['csinodes'], **kwargs)

@tool("k8s_csistoragecapacities", "devops", "Execute k8s_csistoragecapacities command")
def k8s_csistoragecapacities(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['csistoragecapacities'], **kwargs)

@tool("k8s_volumeattachments", "devops", "Execute k8s_volumeattachments command")
def k8s_volumeattachments(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['volumeattachments'], **kwargs)

@tool("k8s_podsecuritypolicies", "devops", "Execute k8s_podsecuritypolicies command")
def k8s_podsecuritypolicies(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['podsecuritypolicies'], **kwargs)

@tool("k8s_podsecurityadmissionreports", "devops", "Execute k8s_podsecurityadmissionreports command")
def k8s_podsecurityadmissionreports(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['podsecurityadmissionreports'], **kwargs)

@tool("k8s_selfsubjectrulesreviews", "devops", "Execute k8s_selfsubjectrulesreviews command")
def k8s_selfsubjectrulesreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['selfsubjectrulesreviews'], **kwargs)

@tool("k8s_subjectrulesreviews", "devops", "Execute k8s_subjectrulesreviews command")
def k8s_subjectrulesreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['subjectrulesreviews'], **kwargs)

@tool("k8s_tokenreviews", "devops", "Execute k8s_tokenreviews command")
def k8s_tokenreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['tokenreviews'], **kwargs)

@tool("k8s_authorizationreviews", "devops", "Execute k8s_authorizationreviews command")
def k8s_authorizationreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['authorizationreviews'], **kwargs)

@tool("k8s_selfsubjectaccessreviews", "devops", "Execute k8s_selfsubjectaccessreviews command")
def k8s_selfsubjectaccessreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['selfsubjectaccessreviews'], **kwargs)

@tool("k8s_subjectaccessreviews", "devops", "Execute k8s_subjectaccessreviews command")
def k8s_subjectaccessreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['subjectaccessreviews'], **kwargs)

@tool("k8s_impersonationreviews", "devops", "Execute k8s_impersonationreviews command")
def k8s_impersonationreviews(**kwargs: Any) -> Dict[str, Any]:
    return _execute("kubectl", ['impersonationreviews'], **kwargs)

@tool("helm_install", "devops", "Execute helm_install command")
def helm_install(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['install'], **kwargs)

@tool("helm_upgrade", "devops", "Execute helm_upgrade command")
def helm_upgrade(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['upgrade'], **kwargs)

@tool("helm_rollback", "devops", "Execute helm_rollback command")
def helm_rollback(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['rollback'], **kwargs)

@tool("helm_uninstall", "devops", "Execute helm_uninstall command")
def helm_uninstall(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['uninstall'], **kwargs)

@tool("helm_list", "devops", "Execute helm_list command")
def helm_list(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['list'], **kwargs)

@tool("helm_status", "devops", "Execute helm_status command")
def helm_status(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['status'], **kwargs)

@tool("helm_history", "devops", "Execute helm_history command")
def helm_history(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['history'], **kwargs)

@tool("helm_test", "devops", "Execute helm_test command")
def helm_test(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['test'], **kwargs)

@tool("helm_template", "devops", "Execute helm_template command")
def helm_template(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['template'], **kwargs)

@tool("helm_lint", "devops", "Execute helm_lint command")
def helm_lint(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['lint'], **kwargs)

@tool("helm_dependency_build", "devops", "Execute helm_dependency_build command")
def helm_dependency_build(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['dependency_build'], **kwargs)

@tool("helm_dependency_update", "devops", "Execute helm_dependency_update command")
def helm_dependency_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['dependency_update'], **kwargs)

@tool("helm_package", "devops", "Execute helm_package command")
def helm_package(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['package'], **kwargs)

@tool("helm_push", "devops", "Execute helm_push command")
def helm_push(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['push'], **kwargs)

@tool("helm_pull", "devops", "Execute helm_pull command")
def helm_pull(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['pull'], **kwargs)

@tool("helm_repo_add", "devops", "Execute helm_repo_add command")
def helm_repo_add(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['repo_add'], **kwargs)

@tool("helm_repo_remove", "devops", "Execute helm_repo_remove command")
def helm_repo_remove(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['repo_remove'], **kwargs)

@tool("helm_repo_list", "devops", "Execute helm_repo_list command")
def helm_repo_list(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['repo_list'], **kwargs)

@tool("helm_repo_update", "devops", "Execute helm_repo_update command")
def helm_repo_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['repo_update'], **kwargs)

@tool("helm_repo_index", "devops", "Execute helm_repo_index command")
def helm_repo_index(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['repo_index'], **kwargs)

@tool("helm_search", "devops", "Execute helm_search command")
def helm_search(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['search'], **kwargs)

@tool("helm_show", "devops", "Execute helm_show command")
def helm_show(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['show'], **kwargs)

@tool("helm_get", "devops", "Execute helm_get command")
def helm_get(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['get'], **kwargs)

@tool("helm_create", "devops", "Execute helm_create command")
def helm_create(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['create'], **kwargs)

@tool("helm_plugin_install", "devops", "Execute helm_plugin_install command")
def helm_plugin_install(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['plugin_install'], **kwargs)

@tool("helm_plugin_list", "devops", "Execute helm_plugin_list command")
def helm_plugin_list(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['plugin_list'], **kwargs)

@tool("helm_plugin_remove", "devops", "Execute helm_plugin_remove command")
def helm_plugin_remove(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['plugin_remove'], **kwargs)

@tool("helm_plugin_update", "devops", "Execute helm_plugin_update command")
def helm_plugin_update(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['plugin_update'], **kwargs)

@tool("helm_plugin_reload", "devops", "Execute helm_plugin_reload command")
def helm_plugin_reload(**kwargs: Any) -> Dict[str, Any]:
    return _execute("helm", ['plugin_reload'], **kwargs)

@tool("terraform_init", "devops", "Execute terraform_init command")
def terraform_init(**kwargs: Any) -> Dict[str, Any]:
    return _execute("terraform", ['init'], **kwargs)

@tool("terraform_plan", "devops", "Execute terraform_plan command")
def terraform_plan(**kwargs: Any) -> Dict[str, Any]:
    return _execute("terraform", ['plan'], **kwargs)

@tool("terraform_apply", "devops", "Execute terraform_apply command")
def terraform_apply(**kwargs: Any) -> Dict[str, Any]:
    return _execute("terraform", ['apply'], **kwargs)

@tool("terraform_destroy", "devops", "Execute terraform_destroy command")
def terraform_destroy(**kwargs: Any) -> Dict[str, Any]:
    return _execute("terraform", ['destroy'], **kwargs)

@tool("terraform_validate", "devops", "Execute terraform_validate command")
def terraform_validate(**kwargs: Any) -> Dict[str, Any]:
    return _execute("terraform", ['validate'], **kwargs)
