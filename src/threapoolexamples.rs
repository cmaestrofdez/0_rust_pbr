use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    ops::Add,
    sync::{
        atomic::{AtomicI64, Ordering},
        Mutex,
    },
    time::Instant,
};

use crate::{primitives::prims::{Ray }, sampler::Sampler, samplerhalton::SamplerHalton, Point2i, Cameras::PerspectiveCamera, raytracerv2::clamp, threapoolexamples1::BBds2i, volumentricpathtracer::volumetric::MediumType};
use crate::{
    imagefilm::{FilterFilm, FilterFilmBox, ImgFilm, Pix},
    materials,
    primitives::Cylinder,
    raytracerv2::interset_scene,
    Bounds2f, Point2f, Point3f, Vector3f,
};
use cgmath::{Deg, InnerSpace, Matrix3, Point2};
use crossbeam::atomic::AtomicCell;
use itertools::Itertools;
use palette::Srgb;
use rayon::prelude::*;
use std::sync::Arc;

pub fn init_tiles(tilesizex: i64, tilesizey: i64, w: i64, h: i64) -> Vec<((f64, f64), (f64, f64))> {
    let num_tiles  = ((h+1) / tilesizey) * ((w+1)/ tilesizex);
    let mut v = vec![((0.,0.), (0.,0.)); num_tiles as usize];
    for y in 0.. (h / tilesizey) {
        for x in 0.. (w / tilesizex) {
            let offset = ((h / tilesizey) * y +x )as usize ;
            let stepsizey = (h / tilesizey) as f64;
            let stepsizex = (w / tilesizex) as f64;
            let xpmin = x as f64 * tilesizex as f64;
            let xpmax = (x as f64 + 1.0) * tilesizex as f64 ;
            let ypmin = y as f64 * tilesizey as f64;
            let ypmax = (y as f64 + 1.0) * tilesizey as f64 ;
            v[offset] = ((xpmin, ypmin), (xpmax, ypmax))
            // v.push(((xpmin, ypmin), (xpmax, ypmax)));
        }
    }
    v
}
pub fn init_tiles_withworkers(num_workers:i64,  w: i64, h: i64) -> Vec<(i64,((i64, i64), (i64, i64)))> {
    if w <=4 || h <=4 {panic!("w and h dims are lower than the dimensions required, perhaps you need set up w = 8 , h = 8 ")}
    // handle w, h. is better  to use power of 2 stuff
    fn handle_dim(x:i64)->f64{
        x as f64
        // let mut  p = x as u64;
        // if !(p).is_power_of_two(){
        //    p.next_power_of_two() as f64
        // }else{
        //     p as f64
        // }
        
    }
    
   let ntilesx = 4_i64;
   let ntilesy = 3_i64;
    let tilesizex = handle_dim(w) as f64 / ntilesx as f64;
     let tilesizey =handle_dim(h) as f64 / ntilesy as f64;
     
     ( 0..num_workers).into_iter().enumerate().map(|(id,i)|{
          let ix =   i % 4;
         let iy =   i  / 4;
      //    println!("{},{}", ix, iy); 
           let x0 = ix as f64 * tilesizex;
         let x1 =  (ix+1) as f64 * tilesizex;
        let y0 =  iy as f64 * tilesizey;
          let y1 =  (iy+1) as f64 * tilesizey;
        
        (id as i64, ((x0 as i64,y0 as i64 ), (x1 as i64 ,y1 as i64)))
    }).collect_vec()
    
 
}
#[test]
pub fn test_tile_gen(){
    
//    [
//     (4,4), (16,16),(32,32),(64,64),
//     (4,64), (16,64),(32,64),(64,64),
//     (128,64), (128,128),(128,256),(128,512),
//     ]
[(128,64)]
    .to_vec().iter().for_each(|res|{
   
    let r  =   init_tiles_withworkers(12,  res.0,  res.1);
    
    let totalarea = BBds2i::new(Point2::new(0 as i64, 0 as i64), Point2::new( res.0 as i64,  res.1 as i64)).area();
    let mut area_temp  = 0 as i64;
        for i in r.iter(){
            println!("{:?}", i.1 );
        area_temp+=  BBds2i::new(Point2::new(i.1.0.0, i.1.0.1), Point2::new(i.1.1.0, i.1.1.1)).area();
    
  
     }
     println!("must be equals : {} , {}", totalarea, area_temp);
     assert_eq!(totalarea, area_temp);
    assert_eq!( r.len() , 12_usize);

   });
    
  
   
  
}

#[derive(Debug, Clone )]
struct Tile {
    pub filterRadius: (f64, f64),
    pub pixels: Vec<Pix>,
    pub dims: Bounds2f,
    pub filterTable: Vec<f64>,
    pub filterTableSize: u32,
    pub w: usize,
    pub h: usize,
    pub filter: FilterFilm,
}

 
struct FilmTilesContainer1 {
 
    
    pub tiles: Arc<Vec<AtomicCell<Tile>>>
}

impl FilmTilesContainer1 {
    pub fn init_tiles() -> Self {
        FilmTilesContainer1 {
          
            tiles:  Arc::new(vec![ AtomicCell::new(Tile::from_filter_box(1024, 1024, (2.0, 2.0))) ]) ,
        }
    }
}

#[derive(Debug, Clone )]
struct FilmTilesContainer {
 
    
    pub tiles: Vec<Tile>,
}
impl FilmTilesContainer {
    pub fn init_tiles() -> Self {
        FilmTilesContainer {
            tiles: vec![Tile::from_filter_box(1024, 1024, (2.0, 2.0))],
        }
    }
    pub fn get(&mut self) -> &mut Tile {
        &mut self.tiles[0]
    }
    pub fn imprimir(&self) {
        println!("imprimir {:?}", self.tiles.len());
    }
}
impl Tile {
    pub fn from_filter_box(w: usize, h: usize, filterRadius: (f64, f64)) -> Self {
        let filterTableSize = 16;

        let pixels: Vec<Pix> = vec![Pix::value_zero(); w * h];
        let mut filterTable: Vec<f64> = vec![0.0; (filterTableSize * filterTableSize) as usize];
        let filterbox = FilterFilm::FilterFilmBoxType(FilterFilmBox::new(filterRadius));

        //  Self::init_pixelbuffer(&pixels);
        //  Self::init_filter_table(& mut filterTable, &filterbox,filterTableSize as usize);

        Tile {
            pixels,
            filterTable,
            filterRadius,
            dims: (Point2f::new(0.0, 0.0), Point2f::new(w as f64, h as f64)),
            w,
            h,
            filter: filterbox,
            filterTableSize,
        }
    }
    pub fn add(&mut self) -> () {
        self.pixels[0] = Pix::value_zero();
    }
}
#[test]
pub fn main_itr() {
    let t = Tile::from_filter_box(1024, 1024, (2.0, 2.0));
    let mut ptr = Arc::new(Mutex::new(ImgFilm::from_filter_gauss(
        1024 as usize,
        1024 as usize,
        (2.0, 2.0),
        1.0,
    )));

    let mut tileshared_ptr = Arc::new(Mutex::new(t));
    // arc clone es especial. copia una referencia y la cuenta almacenandola
    let mut ptr_clone = tileshared_ptr.clone();
    let mut film = RefCell::new(ImgFilm::from_filter_gauss(
        1024 as usize,
        1024 as usize,
        (2.0, 2.0),
        1.0,
    ));
    init_tiles(512, 512, 1024, 1024)
        .par_iter()
        .for_each(move |x| {
            // f.borrow_mut(). add_sample(&Point2f::new(1.0,1.0),Srgb::new(0.0,0.0,0.0));

            // (&mut *tileshared.borrow_mut()).doSome();
            // ptr_clone.lock().unwrap().doSome();
            println!("{:?}", x);
            ptr.lock()
                .unwrap()
                .add_sample(&Point2f::new(1.0, 1.0), Srgb::new(0.0, 0.0, 0.0));
        });

    // tileshared_ptr.lock().unwrap().doSome();
}

#[test]
pub fn main_film_itr() {
    let mut ptr = Arc::new(Mutex::new(FilmTilesContainer::init_tiles()));

    let mut ptr_clone = ptr.clone();
    init_tiles(32, 32, 1024, 1024)
        .par_iter()
        .for_each(move |x| {
            println!("{:?}", x);
            let mut tile = ptr_clone.lock().unwrap();
            let b = tile.borrow_mut();
            let mut t = b.get();
            t.add();
        });
    let tile = ptr.lock().unwrap().imprimir();
}
pub fn main_crossbeam() {
    let mut ptr = Arc::new(FilmTilesContainer::init_tiles());
    let num_cores = num_cpus::get();
    crossbeam::scope(|scope| {
        let (sx, rx) = crossbeam_channel::bounded(num_cores);
        //producer thread
        scope.spawn(move |_| {
            ptr.imprimir();
            let sxc = sx.clone();
            for i in 0..100 {
                sx.send(i).unwrap();
            }
            drop(sxc); // una vez hemos hecho el for este sender lo eliminamos
        });

        for _ in 0..num_cores {
            let rxc = rx.clone();
            // toca los workers
            scope.spawn(move |_| {
                for x in rxc.iter() {
                    println!("{:?}", x);
                }
            });
        }
        drop(rx);
    })
    .unwrap();
}

pub fn main_crossbeam_intersection() {
    let mat3rot = Matrix3::from_angle_z(Deg(90.0));
    let start = Instant::now();
    let c = Arc::new(Cylinder::new(
        Vector3f::new(0.0, 0.0, 0.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    ));
    let mut ptr = Arc::new(FilmTilesContainer::init_tiles());
    let num_cores = num_cpus::get();
    let mut cnt = 0;
    crossbeam::scope(|scope| {
        let (sx, rx) = crossbeam_channel::bounded(num_cores);
        //producer thread
        scope.spawn(move |_| {
            let sxc = sx.clone();
            let numitec = 1000000000;

            for i in 0..numitec {
                let fi = 2.0 * (i as f64 / numitec as f64) - 1.0;
                let r = Ray::<f64> {  medium:None , is_in_medium: false,
                    origin: Point3f::new(-2.0, 0.1, fi),
                    direction: Vector3f::new(1.0, 0.0, 0.0).normalize(),
                };
                sx.send(r).unwrap();
            }
            drop(sxc); // una vez hemos hecho el for este sender lo eliminamos
        });

        for _ in 0..num_cores {
            let rxc = rx.clone();
            let cyl = &c;
            // toca los workers
            scope.spawn(move |_| {
                for r in rxc.iter() {
                    if let Some(hit) = cyl.intersect(&r, 0.00001, f64::MAX) {}
                }
            });
        }
        drop(rx);
    })
    .unwrap();
    println!("cnt {}", cnt);
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection() {
    let mat3rot = Matrix3::from_angle_z(Deg(90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 0.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let start = Instant::now();
    let mut cnt = 0;
    let numitec = 1000000000;
    for i in 0..numitec {
        let fi = 2.0 * (i as f64 / numitec as f64) - 1.0;

        if let Some(hit) = c.intersect(
            &Ray::<f64> {  medium:None,  is_in_medium: false,
                origin: Point3f::new(-2.0, 0.1, fi),
                direction: Vector3f::new(1.0, 0.0, 0.0).normalize(),
            },
            0.00001,
            f64::MAX,
        ) {
            cnt += 1;
        } else {
            println!("no intersection");
        }
    }
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
}
pub fn main_test_intersection_first_box() {
    println!("main_test_intersection_first_box");
    let mut cnt = 0;
    let numitec = 1000000000;
    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let bf3 = c.world_bound();

    let start = Instant::now();
    for i in 0..numitec {
        let fi = 2.0 * (i as f64 / (numitec) as f64) - 1.0;
        let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
        let r = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            Vector3f::new(sc.0, 0.0, sc.1).normalize(),
        );
        let hitsbox = bf3.intersect(&r, 0.00001, std::f64::MAX);
        if hitsbox.0 {
            let hits = c.intersect(&r, 0.00001, std::f64::MAX);
            if let Some(nehi) = hits {
                cnt += 1;
            }
        }
    }

    println!("{}", cnt);
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection_second_box() {
    println!("main_test_intersection_second_box only test bounding box");
    let mut cnt = 0;
    let numitec = 1000000000;
    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let bf3 = c.world_bound();

    let start = Instant::now();
    for i in 0..numitec {
        let fi = 2.0 * (i as f64 / (numitec) as f64) - 1.0;
        let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
        let r = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            Vector3f::new(sc.0, 0.0, sc.1).normalize(),
        );
        let hitsbox = bf3.intersect(&r, 0.00001, std::f64::MAX);

        if hitsbox.0 {
            cnt += 1;
        }
    }

    println!("{}", cnt);
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection_third_box() {
    println!("main_test_intersection_second_box only test primitives interface box");
    let mut cnt = 0;
    let numitec = 1000000000;

    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let bf3 = c.world_bound();

    let start = Instant::now();
    for i in 0..numitec {
        let fi = 2.0 * (i as f64 / (numitec) as f64) - 1.0;
        let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
        let r = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            Vector3f::new(sc.0, 0.0, sc.1).normalize(),
        );
        let hits = c.intersect(&r, 0.00001, std::f64::MAX);
        if let Some(nehi) = hits {
            cnt += 1;
        }
    }

    println!("{}", cnt);
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
}

fn createRaysTest(nrays: usize) -> Vec<Ray<f64>> {
    let startbunchofrays = Instant::now();
    let mut bunchofrays: Vec<Ray<f64>> = Vec::with_capacity(nrays);
    unsafe {
        bunchofrays.set_len(nrays);
    }
    for i in 0..nrays {
        let fi = 2.0 * (i as f64 / (nrays) as f64) - 1.0;
        let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
        let r = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            Vector3f::new(sc.0, 0.0, sc.1).normalize(),
        );
        bunchofrays[i] = r;
    }
    // println!("      startbunchofrays time: {} seconds", startbunchofrays.elapsed().as_secs_f64());
    return bunchofrays;
}
pub fn main_test_intersection_four_par_item() {
    println!("main_test_intersection_four_par_item ");
    let mut cnt = 0;
    let arcnt = Arc::new(cnt);
    let atoCnt = AtomicI64::new(0);
    let numitec = 100000000;

    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let bf3 = c.world_bound();

    let start = Instant::now();
    let startbunchofrays = Instant::now();
    let mut bunchofrays: Vec<Ray<f64>> = Vec::with_capacity(numitec);
    unsafe {
        bunchofrays.set_len(numitec);
    }
    for i in 0..numitec {
        let fi = 2.0 * (i as f64 / (numitec) as f64) - 1.0;
        let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
        let r = Ray::new(
            Point3f::new(0.0, 0.0, 0.0),
            Vector3f::new(sc.0, 0.0, sc.1).normalize(),
        );
        bunchofrays[i] = r;
    }
    println!(
        "      startbunchofrays time: {} seconds",
        startbunchofrays.elapsed().as_secs_f64()
    );

    let startbunchofraysintersection = Instant::now();
    bunchofrays.par_iter().for_each(move |r| {
        let hits = c.intersect(r, 0.00001, std::f64::MAX);
        if let Some(nehi) = hits {
            // print!(".");
            atoCnt.fetch_add(1, Ordering::Relaxed);
        }
    });
    println!(
        "   \n   startbunchofrays intersection time: {} seconds",
        startbunchofraysintersection.elapsed().as_secs_f64()
    );

    println!("total time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection_5_par_item() {
    println!("main_test_intersection_5_par_item ");
    let mut cnt = 0;
    let arcnt = Arc::new(cnt);
    let atoCnt = AtomicI64::new(0);
    let numitec = 100000000;

    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c = Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    );
    let bf3 = c.world_bound();

    let start = Instant::now();
    let startbunchofrays = Instant::now();
    init_tiles(64, 64,512, 512).par_iter().for_each(|tile| {
        let starttilerendering = Instant::now();
        //     println!("{:?}",tile);
        for px in itertools::iproduct!(tile.0 .0 as i32..tile.1 .0 as i32,tile.0 .1 as i32..tile.1 .1 as i32) {
            let starthitsrays = Instant::now();
            
          
            for r in createRaysTest(512*10*10).iter() {
                let bound = c.world_bound();
                let ishit = bound.intersect(r, 0.00001, std::f64::MAX);
                if ishit .0 == true {
                    let hits = c.intersect(r, 0.00001, std::f64::MAX);
                    if let Some(newhit) = hits {
                        // println!("{:?}",  newhit.point);
                        atoCnt.fetch_add(1, Ordering::Relaxed);
                    }
                 }
               
// tengo que probar el area disk light
// puedo hacer una variante de scene interset_scene  con bbox
            }
            // println!("starthitsrays total time: {} seconds", starthitsrays.elapsed().as_secs_f64());
        }
        println!(" tile {:?} ", tile);
         println!("starttilerendering total time: {} seconds  tile {:?} ", starttilerendering.elapsed().as_secs_f64(), tile);
    });

    println!("total time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection_6_crossbeam(){
   
    
    let spp = 512_u64;
    let numrayspersample = 3*3;
    let dims = 512*512;
    let totalnumyars = spp * numrayspersample*dims;
    println!("main_test_intersection_6_crossbeam total number of rays evaluated: {}", totalnumyars);
    let num_cores = num_cpus::get();
    
    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c =  Arc::new(Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    ));
   
    // let bf3 = c.world_bound();

    let start = Instant::now();
    crossbeam::scope(|scope| {
        let (sx, rx) = crossbeam_channel::bounded(num_cores);
        scope.spawn(move |_| {
            let piperays = sx.clone();
            for bactch in 0..512*512{
                let it =  (bactch,createRaysTest((spp*numrayspersample) as usize) ) ;
                piperays.send(it).unwrap ();
            }
           
            drop(piperays);
            
        });
        for _ in 0..num_cores {
            let rxc = rx.clone();
            let cyl = &c;
            let cnt = AtomicI64::new(0);
            scope.spawn(move |_| {
             
               for batchrays in rxc.iter(){
                  
                   //  println!("batch {}  ", batchrays.0 );
                    for r in batchrays.1 {
                        let bound =cyl.world_bound();
                        if true == bound.intersect(&r, 0.00001, f64::MAX).0{
                            // print!(".");
                            cnt.fetch_add(1, Ordering::AcqRel);
                        }
                    }
                  
               } 
               print!("batch {:?}  \n", cnt );
            });
        }
        drop(rx);
    }).unwrap();
    println!("\ntotal time: {} seconds", start.elapsed().as_secs_f64());
}

pub fn main_test_intersection_6_serial_way(){
    let spp = 512_u64;
    let numrayspersample = 3*3;
    let dims = 512*512;
    let totalnumyars = spp * numrayspersample*dims;
    println!("main_test_intersection_6_serial_way: {}", totalnumyars);
    let num_cores = num_cpus::get();
    let cnt = AtomicI64::new(0);
    let mat3rot = Matrix3::from_angle_x(Deg(-90.0));
    let c =  Arc::new(Cylinder::new(
        Vector3f::new(0.0, 0.0, 3.0),
        mat3rot,
        -1.0,
        1.0,
        1.0,
        materials::MaterialDescType::NoneType,
    ));
    let start = Instant::now();
    for bactch in 0..512*512{
        let it =  (bactch,createRaysTest((spp*numrayspersample) as usize) ) ;
        for r in it.1.iter(){
                  
            let bound =c.world_bound();
            if true == bound.intersect(&r, 0.00001, f64::MAX).0{
                // print!(".");
                cnt.fetch_add(1, Ordering::AcqRel);
            }
           
        } 
    }
    println!("\ntotal time: {} seconds", start.elapsed().as_secs_f64());
}



pub fn main_test_intersection_7_test(){
   let spp = 8;
   let num_cores = num_cpus::get();
   let filmres = (255, 255);
    let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, filmres);
    let mut filmtilecontainer = Arc::new(  FilmTilesContainer1::init_tiles());
 
    let mut samplerhalton  :  Box<dyn Sampler> = Box::new(
        SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(filmres.0 as i64, filmres.1 as i64),spp,false,));


    crossbeam::scope(|scope| {
       let tiles =  filmtilecontainer.tiles.clone();
       for t  in tiles.iter()  {
          let at =  t.clone();
          
       }
        // let (sx, rx) = crossbeam_channel::bounded(num_cores);
        // let camera = &cameraInstance; 
        // let sampler = &samplerhalton;
        // let  filmcontainerptr =  Arc::clone(&filmtilecontainer);
        
        // scope.spawn(move |_| {
        //   let tiles =   init_tiles(32, 32, 512, 512);;
        //   let   sxclone = sx.clone();
           
        //   for tile in tiles. iter(){
        //     sxclone.send(tile.clone()).unwrap();
        //   }
        // drop(sxclone);
        // });
        // for _ in 0..num_cores { 
        //     scope.spawn(move |_|{
        //         let mut f= &mut  filmcontainerptr.tiles.get(0).unwrap();
                
        //     });

        // }
       
       
        // for _ in 0..num_cores { 
        //     let   rxclone = rx.clone();
        //     scope.spawn(move |_| {
               
        //         let tilerender  = rxclone.iter();
        //         for ttile in tilerender{
        //             filmcontainerptr.get();
        //             // filmtile =  film.get_tile(ttile);
        //             // for px in   filmres.bounds().iter(){
        //             //     sampler.start_pixel(Point2i::new(xx as i64, yy as i64));
        //             // }
        //             // pipe.send(filmtile);
        //             // ;
                    
        //         }
              
        //       });
        // };

        // recibimos 
  
          

    }).unwrap();
}