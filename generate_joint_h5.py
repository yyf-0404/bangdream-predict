"""Build/audit V2 event surfaces and joint weight plans. No training imports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from config import CURVE_DATA, CURVE_SPLIT, Config
from data import raw
from data.event_split import load_event_split
from data.joint_dataset import JointDataset, prepare_joint_artifact, save_joint_hdf5
from data.joint_sequences import build_joint_event
from data.rewards import DEFAULT_REWARD_TABLE, RewardTable


def main():
    parser = argparse.ArgumentParser(description="生成/审计 V2 联合累计数据与 1:1 权重，不训练")
    parser.add_argument('--raw_dir', default=None)
    parser.add_argument('--out_h5', default=None)
    parser.add_argument('--split_manifest', default=None)
    parser.add_argument('--extend-validation-from', metavar='H5',
                        help='沿用既有快照中的活动、训练划分和权重配置，只从 raw 读取新增验证活动')
    parser.add_argument('--reward_table', default=str(DEFAULT_REWARD_TABLE))
    parser.add_argument('--t_min', type=int, default=None)
    parser.add_argument('--audit_only', action='store_true', help='读取全部曲线并核对联合样本/权重，不写 HDF5')
    parser.add_argument('--overwrite', action='store_true', help='显式允许替换指定 V2 输出文件')
    args = parser.parse_args()
    cfg = Config()
    root = args.raw_dir or cfg.raw_dir
    manifest = load_event_split(args.split_manifest or CURVE_SPLIT)
    with open(f'{root}/events_all.json', encoding='utf-8') as handle:
        metadata = json.load(handle)
    with open(f'{root}/characters_main.json', encoding='utf-8') as handle:
        characters = json.load(handle)
    manifest.validate_metadata(metadata)
    now = int(datetime.now(timezone.utc).timestamp() * 1000)
    rewards = RewardTable(args.reward_table)
    target = args.out_h5 or CURVE_DATA
    options = dict(t_min=cfg.t_min if args.t_min is None else args.t_min,
                   weight_mode=cfg.weight_mode, weight_floor=cfg.weight_floor,
                   tier_center=cfg.tier_center, tier_alpha=cfg.tier_alpha,
                   tier_floor=cfg.tier_floor, tier_1500_weight=cfg.tier_1500_weight)
    frozen = {}
    if args.extend_validation_from:
        if not args.audit_only and Path(target).resolve() == Path(args.extend_validation_from).resolve():
            parser.error('扩展验证集需要单独的输出文件')
        train = JointDataset(args.extend_validation_from, 'train')
        val = JointDataset(args.extend_validation_from, 'val')
        if (train.event_ids != manifest.training_ids or not val.event_ids < manifest.validation_ids
                or train.event_split_report['manifest']['server'] != manifest.server
                or train.event_split_report['version'] != manifest.manifest.get('parent_version')):
            raise ValueError('验证集扩展必须匹配父版本，保留所有训练和验证活动，并新增验证活动')
        if train.reward_hash != rewards.sha256:
            raise ValueError('奖励表与父快照不一致')
        options = {key: train.weight_report[key] for key in options}
        if (any(val.weight_report[key] != value for key, value in options.items())
                or args.t_min is not None and args.t_min != options['t_min']):
            raise ValueError('切点或权重配置与父快照不一致')
        frozen = {event_id: source for source in (train, val) for event_id in source.event_ids}
    events = []
    for event_id in sorted(manifest.event_ids):
        if event_id in frozen:
            events.append(frozen[event_id].event(event_id))
            continue
        end = raw.get_end_at(manifest.server, event_id, metadata=metadata)
        if end is None or end > now:
            raise ValueError(f'清单活动未结束或缺少时间: {event_id}')
        events.append(build_joint_event(
            event_id, server=manifest.server, raw_dir=root, metadata=metadata, characters=characters,
            reward_table=rewards, event_split=manifest, current_time_ms=now,
        ))
        if len(events) % 16 == 0:
            print(f'已核对联合曲线 {len(events)}/{len(manifest.event_ids)}', flush=True)
    if args.audit_only:
        _, plans, norm = prepare_joint_artifact(events, manifest, **options)
        reports = {split: plan.report for split, plan in plans.items()}
        print('训练集累计归一化:', json.dumps(norm, ensure_ascii=False))
    else:
        reports = save_joint_hdf5(events, target, event_split=manifest, reward_table=rewards,
                                 overwrite=args.overwrite, **options)
        print(f'V2 数据已保存至 {target}；没有启动模型或训练。')
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    if args.audit_only:
        print('联合数据、奖励表和权重审计通过；未写入 HDF5、未训练。')


if __name__ == '__main__':
    main()
