import torchreid
torchreid.models.show_avai_models()
datamanager = torchreid.data.ImageDataManager(
    root="reid-data",
    sources="market1501pose",
    targets="market1501pose",
    height=256,
    width=128,
    batch_size_train=32,
    batch_size_test=32,
    transforms=["random_flip", "random_crop"],
    workers=8
)


model = torchreid.models.build_model(
    name="osnetmod_ain_x0_25",
    num_classes=datamanager.num_train_pids,
    loss="triplet",
    pretrained=True
)

model = model.cuda()

optimizer = torchreid.optim.build_optimizer(
    model,
    optim="adam",
    lr=0.0003
)

scheduler = torchreid.optim.build_lr_scheduler(
    optimizer,
    lr_scheduler="single_step",
    stepsize=30
)

engine = torchreid.engine.ImageTripletEnginePose(
    datamanager,
    model,
    optimizer=optimizer,
    scheduler=scheduler,
    label_smooth=True
)

engine.run(
    save_dir="log/osnetmod_ain_x0_25_part_vis",
    max_epoch=120,
    eval_freq=10,
    print_freq=10,
    test_only=False)