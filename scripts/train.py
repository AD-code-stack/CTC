    work_dir.mkdir(parents=True, exist_ok=True)
    save_json(work_dir / 'token_map.json', token_map)
    
    # 保存词频统计
    save_json(work_dir / 'token_frequency.json', 
               {k: v for k, v in token_counter.most_common(100)})

    # CTC损失
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['train']['lr'],
        weight_decay=config['train']['weight_decay'],
    )
    
    # 学习率调度（参考TFNet：MultiStepLR）
    epochs = int(config['train']['epochs'])
    scheduler_type = config['train'].get('scheduler', 'none')
    scheduler = None
    
    if scheduler_type == 'multisteplr':
        milestones = config['train'].get('milestone_epochs', [30, 50])
        gamma = config['train'].get('gamma', 0.2)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
        print(f'使用MultiStepLR调度器: milestones={milestones}, gamma={gamma}')
    elif scheduler_type == 'cosine':
        min_lr = config['train'].get('min_lr', 0.00001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=min_lr
        )
        print(f'使用CosineAnnealingLR调度器: eta_min={min_lr}')

    best_val_ter = float('inf')
    patience = config['train'].get('patience', 20)
    patience_counter = 0
    history: list[dict[str, Any]] = []

    print(f'\n开始训练 (最多 {epochs} epochs, patience={patience})...\n')

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        
        train_metrics, _ = _run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
        val_metrics, val_preview = _run_epoch(model, val_loader, criterion, device, optimizer=None)
        
        train_metrics.epoch = epoch