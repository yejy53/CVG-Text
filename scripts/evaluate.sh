# Satellite
python zeroshot.py --version NewYork-mixed --img_type sat  --model CLIP-L/14@336 --expand
# OSM
python zeroshot.py --version NewYork-mixed --img_type OSM  --model CLIP-L/14@336 --expand
# Brisbane Pano-view
python zeroshot.py --version Brisbane --img_type OSM  --model CLIP-L/14@336 --expand
# Clip-B
python zeroshot.py --version Brisbane --img_type OSM  --model CLIP-B/16
# Clip-L (Unexpand) on Tokyo Single-view Texts
python zeroshot.py --version Tokyo-photos --img_type sat  --model CLIP-L/14@336
# Evaluate the checkpoint trained by mixed(Pano + Sing-view) texts on just Single-view texts
python zeroshot.py --version NewYork-photos --img_type osm  --model CLIP-L/14@336 --expand --checkpoint ./checkpoints/CLIP-L_14@336/long_model_NewYork-mixed_1e-05_128_osm_epoch30_59.08.pth
