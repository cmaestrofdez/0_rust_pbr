use crate::{
    materials::{MaterialDescType, Plastic},
    primitives::PrimitiveType,
    Config,
    Lights::Light,
    Lights::PointLight,
    Spheref64, Vector3f,
};
use cgmath::{Point2, Point3};
use palette::Srgb;

use crate::{
    bdpt2::bdpt2::PathTypes,
    imagefilm::{FilterFilm, FilterFilmGauss},
    sampler::{Sampler, SamplerType, SamplerUniform},
    scene::Scene1,
    threapoolexamples1::{pararell, BBds2f, BBds2i},
    Cameras::PerspectiveCamera,
    Point3f,
};

use self::metropolistransport::{
    mlt_camera_path, mlt_light_path, MetropolisIntegrator, SamplerMetropolis, TraitMltSampler,
    CHANNELS,
};

pub mod metropolistransport {
    use crate::{Lum, Point2f};
    use core::panic;
    use std::borrow::{BorrowMut, Cow};
    use std::cell::{RefCell, RefMut};
    use std::fmt::Debug;
    use std::ops::Deref;
    use std::rc::Rc;
    use std::sync::{Arc, RwLock};
    use std::time::Instant;

    use cgmath::{Deg, InnerSpace, Matrix3, Point2, Point3, Vector1, Vector3};
    use float::Sqrt;
    use num_traits::Float;
    use palette::Srgb;
    use rayon::prelude::{IntoParallelRefIterator, ParallelIterator, IntoParallelIterator};

    use crate::bdpt2::bdpt2::{
        bidirectional, cameraTracerStrategy, cameraTracerStrategy1, compute_weights1,
        convert_static, emissionDirectStrategy, emissionDirectStrategy1, init_camera_path,
        init_light_path, lightTracerStrategy, lightTracerStrategy1, BdptVtx, Mode, PathCameraVtx,
        PathLightVtx, PathTypes, PathVtx, PlotChains,
    };
    use crate::imagefilm::{FilterFilm, FilterFilmGauss};
    use crate::integrator::{isBlack, IntergratorBaseType};
    use crate::materials::{Fr, IsSpecular, MaterialDescType, Pdf, Plastic, RecordSampleIn};
    use crate::primitives::prims::HitRecord;
    use crate::primitives::prims::{IntersectionOclussion, IsEmptySpace, Ray};
    use crate::primitives::{Cylinder, Disk, PrimitiveType};
    use crate::raytracerv2::{
        clamp, interset_scene_bdpt, new_interface_for_intersect_scene__bdpt,
        new_interface_for_intersect_scene__bdpt_walklight,
    };
    use crate::sampler::custom_rng::CustomRng;
    use crate::sampler::{Sampler, SamplerType};
    use crate::samplerhalton::SamplerHalton;
    use crate::scene::Scene1;
    use crate::threapoolexamples1::{pararell, BBds2f, BBds2i};
    use crate::Cameras::{PerspectiveCamera, RecordSampleImportanceIn};
    use crate::Lights::{
        Distribution1d, GetEmission, IsAreaLight, Light, LigtImportanceSampling,
        New_Interface_AreaLight, PdfEmission, PointLight, RecordPdfEmissionIn,
        RecordSampleLightIlumnation,
    };
    use crate::{Point2i, Point3f, Spheref64, Vector3f};
    // use crate::raytracerv2::{Scene, interset_scene, interset_scene_primitives, interset_scene_bdpt};

    use crate::materials::SampleIllumination;
    use crate::Lights::RecordSampleLightEmissionIn;
    use crate::Lights::RecordSampleLightEmissionOut;
    use crate::Lights::SampleEmission;

    use crate::Lights::IsBackgroundAreaLight;
    use crate::Lights::SampleLightIllumination;

    use crate::Config;

    pub fn walklight_1<S: Sampler>(
        r: &Ray<f64>,
        paths: &mut Vec<PathTypes>,
        beta: Srgb,
        pdfdir: f64,
        scene: &Scene1,
        sampler: &mut S,
        mode: Mode,
        maxdepth: usize,
    ) -> usize {
        if (maxdepth == 0) {
            return 0;
        };
        let mut transport = beta;
        let mut pdfnext = pdfdir;
        let mut pdfrev = 0.0;
        let mut rcow = Cow::Borrowed(r);
        let mut depth: usize = 1;
        while true {
            let tr = rcow.clone();
            // let (mut hitop, light ) = interset_scene_bdpt(&tr, scene);
            let (mut hitop, lightid) =
                new_interface_for_intersect_scene__bdpt_walklight(&tr, scene);

            if let Some(hiit) = hitop {
                let mut hitcow = Cow::Borrowed(&hiit);
                paths.push(PathTypes::PathVtxType(PathVtx::new_pathvtx_wilthlightid(
                    *hitcow, transport, pdfnext, 0.0, mode, lightid,
                )));

                let newpdf = convert_static(pdfnext, &paths[depth - 1], &paths[depth], scene);
                (&mut paths[depth]).set_pdfnext(newpdf);
                if depth >= maxdepth {
                    return depth + 1;
                }
                let psample = sampler.get2d();
                let recout = hitcow
                    .material
                    .sample(RecordSampleIn::from_hitrecord(*hitcow, &rcow, psample));
                if recout.checkIfZero() {
                    break;
                }
                transport = recout.compute_transport(transport, &hitcow);
                pdfrev = hitcow.material.pdf(recout.next, hitcow.prev.unwrap());
                pdfnext = recout.pdf;

                let newr = Ray::<f64>::new(hitcow.point, recout.next);
                rcow = Cow::Owned(newr);
            } else {
                match mode {
                    Mode::from_camera => {
                        paths.push(PathTypes::PathLightVtxType(PathLightVtx::from_endpoint(
                            *tr, transport, pdfnext,
                        )));
                    }
                    Mode::from_light => {}
                }
                //
                break;
            }
            let updatepdf = convert_static(pdfrev, &paths[depth], &paths[depth - 1], scene);
            (&mut paths[depth - 1]).set_pdfrev(updatepdf);
            depth = depth + 1;
        }
        depth
    }
    pub fn mlt_merge_path<S: Sampler>(
        scene: &Scene1,
        s_light: i32, //  s_light
        t_camera: i32,
        pathcamera: &mut Vec<PathTypes>,
        pathlight: &mut Vec<PathTypes>,
        sampler: &mut S,
        cameraInstance: &PerspectiveCamera,
        pfilmRaster: &(f64, f64),
    ) -> Srgb {
        if (s_light == 1 && t_camera == 2) {
            // println!("");
        }
        let mut Lsum = Srgb::new(0.0, 0.0, 0.0);

        let mut L_TOTAL = Srgb::new(0.0, 0.0, 0.0);
        let mut L_bdir_TOTAL = Srgb::new(0.0, 0.0, 0.0);
        let mut L_cam_TOTAL = Srgb::new(0.0, 0.0, 0.0);
        let mut L_lightStra_TOTAL = Srgb::new(0.0, 0.0, 0.0);
        let mut ckcsumpdTOTAL = 0.0;

        let mut Llightstrategy = Srgb::new(0.0, 0.0, 0.0);
        let mut Lcamerastrategy = Srgb::new(0.0, 0.0, 0.0);
        let mut LbirectionalStrategy = Srgb::new(0.0, 0.0, 0.0);
        let mut L_PX_TOTAL = Srgb::new(0.0, 0.0, 0.0);

        let depth: i32 = s_light + t_camera - 2;

        // let pfilm = Point3f::new(pfilm.0, pfilm.1,0.0);
        if t_camera == 1 && s_light == 1 {
            return Srgb::new(0., 0., 0.);
        }

        let vtxcamera = &pathcamera[t_camera as usize - 1];

        let islightvtx = match vtxcamera {
            PathTypes::PathLightVtxType(l) => true,
            _ => false,
        };
        if t_camera > 1 && s_light != 0 && islightvtx {
            return Srgb::new(0., 0., 0.);
        };

        if s_light == 0 {
            if true {
                let L = emissionDirectStrategy1(t_camera, scene, sampler, &pathlight, &pathcamera);
                if !isBlack(L) {
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight = compute_weights1(
                        s_light, t_camera, scene, sampledvtx, pathlight, pathcamera,
                    );
                    Lsum = LigtImportanceSampling::addSrgb(
                        LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                        Lsum,
                    );
                }
            }
        } else if t_camera == 1 {
            if false {
                let (L, sampledvtx) = cameraTracerStrategy1(
                    s_light,
                    scene,
                    &Point3f::new(pfilmRaster.0, pfilmRaster.1, 0.0),
                    sampler,
                    *cameraInstance,
                    &pathlight,
                    &pathcamera,
                );

                if !isBlack(L) {
                    let weight = compute_weights1(
                        s_light,
                        t_camera,
                        scene,
                        sampledvtx.unwrap(),
                        pathlight,
                        pathcamera,
                    );
                    let Lcam = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                    L_cam_TOTAL = LigtImportanceSampling::addSrgb(L_cam_TOTAL, Lcam);
                    //  println!("     Lcam  {:?}", Lcam);
                    Lcamerastrategy = LigtImportanceSampling::addSrgb(Lcamerastrategy, Lcam);
                    Lsum = LigtImportanceSampling::addSrgb(Lsum, Lcam);
                }
            }
        } else if s_light == 1 {
            if true {
                let (L, vtx) =
                    lightTracerStrategy1(t_camera, scene, sampler, &pathlight, &pathcamera);

                if !isBlack(L) {
                    //  tengo que hacer probar las novedades!
                    let weight = compute_weights1(
                        s_light,
                        t_camera,
                        scene,
                        vtx.unwrap(),
                        pathlight.as_mut_slice(),
                        pathcamera.as_mut_slice(),
                    );
                    // println!("weight {:?}", weight);
                    let LPath = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                    Llightstrategy = LigtImportanceSampling::addSrgb(Llightstrategy, LPath);
                    L_lightStra_TOTAL = LigtImportanceSampling::addSrgb(L_lightStra_TOTAL, LPath);
                    Lsum = LigtImportanceSampling::addSrgb(
                        LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                        Lsum,
                    );
                }
            }
        } else {
            if true {
                let Lbidirectional = bidirectional(
                    s_light,
                    t_camera,
                    scene,
                    &Point3f::new(pfilmRaster.0, pfilmRaster.1, 0.0),
                    &pathlight,
                    &pathcamera,
                );
                if !isBlack(Lbidirectional) {
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight = compute_weights1(
                        s_light, t_camera, scene, sampledvtx, pathlight, pathcamera,
                    );
                    let LPath =
                        LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32);
                    Lsum = LigtImportanceSampling::addSrgb(LPath, Lsum);
                }
            }
        }

        Lsum
    }
    pub fn mlt_light_path<S: Sampler>(
        scene: &Scene1,
        paths: &mut Vec<PathTypes>,
        sampler: &mut S,
        maxdepth: usize,
    ) -> usize {
        // sample light emision,get first ray
        // call walk
        if maxdepth == 0 {
            return 0;
        }

        let l = &scene.lights[0];
        let pdflightselect = 1.0;
        let u0 = sampler.get2d();
        let u1 = sampler.get2d();

        let recout = l.sample_emission(RecordSampleLightEmissionIn::from_sample2dx2d(u0, u1));
        let a = recout.n.dot(recout.ray.unwrap().direction).abs()
            / (recout.pdfdir * recout.pdfpos * pdflightselect);
        let beta = LigtImportanceSampling::mulScalarSrgb(recout.Lemission, a as f32);

        let lightVtx = PathTypes::PathLightVtxType(PathLightVtx::from_hitrecord_lightid(
            &l,
            HitRecord::from_point(recout.ray.unwrap().origin, recout.n),
            recout.Lemission,
            recout.pdfpos * pdflightselect,
            0,
        ));
        paths.push(lightVtx);

        let ra = Ray::new(Point3f::new(0., 0., 0.), Vector3f::new(0.0, 0.0, 1.0));
        walklight_1(
            &recout.ray.unwrap(),
            paths,
            beta,
            recout.pdfdir,
            scene,
            sampler,
            Mode::from_light,
            maxdepth - 1,
        ) + 1
    }

    pub fn mlt_camera_path<S: Sampler>(
        scene: &Scene1,
        pfilm: &(f64, f64),
        paths: &mut Vec<PathTypes>,
        camera: PerspectiveCamera,
        sampler: &mut S,
        search_depth: usize,
    ) -> usize {
        let beta = Srgb::new(1.0, 1.0, 1.0);
        let r = camera.get_ray(pfilm);
        let (pdfpos, pdfdir) = camera.pdfWemission(&r);

        let cameravtx = PathTypes::PathCameraVtxType(PathCameraVtx::new_camerapath_init(camera, r));
        paths.push(cameravtx);

        let depth = walklight_1(
            &r,
            paths,
            beta,
            pdfdir,
            scene,
            sampler,
            Mode::from_camera,
            search_depth - 1,
        );
        depth
    }

    // ref : https://github.com/wahn/rs_pbrt/blob/d016b75e57391c9989332a18dbdcd11ec63e0d3c/src/core/pbrt.rs#L270
    pub fn erf_inv(x: f64) -> f64 {
        let clamped_x = clamp(x, -0.99999, 0.99999);
        let mut w = -((1.0 - clamped_x) * (1.0 + clamped_x)).ln();
        let mut p;
        if w < 5.0 {
            w -= 2.5;
            p = 2.810_226_36e-08;
            p = 3.432_739_39e-07 + p * w;
            p = -3.523_387_7e-06 + p * w;
            p = -4.391_506_54e-06 + p * w;
            p = 0.000_218_580_87 + p * w;
            p = -0.001_253_725_03 + p * w;
            p = -0.004_177_681_640 + p * w;
            p = 0.246_640_727 + p * w;
            p = 1.501_409_41 + p * w;
        } else {
            w = w.sqrt() - 3.0;
            p = -0.000_200_214_257;
            p = 0.000_100_950_558 + p * w;
            p = 0.001_349_343_22 + p * w;
            p = -0.003_673_428_44 + p * w;
            p = 0.005_739_507_73 + p * w;
            p = -0.007_622_461_3 + p * w;
            p = 0.009_438_870_47 + p * w;
            p = 1.001_674_06 + p * w;
            p = 2.832_976_82 + p * w;
        }
        p * clamped_x
    }

    #[derive(Debug, Clone)]
    pub struct MetropolisSample {
        pub value: f64,
        backup: f64,
        last_iteration: i64,
        modified: i64,
    }
    impl Default for MetropolisSample {
        fn default() -> Self {
            MetropolisSample {
                value: 0.,
                backup: 0.,
                last_iteration: 0,
                modified: 0,
            }
        }
    }
    impl MetropolisSample {
        pub fn new() -> Self {
            Self::default()
        }
        pub fn store(&mut self) {
            self.backup = self.value;
            self.modified = self.last_iteration;
        }
        pub fn rollback(&mut self) {
            self.value = self.backup;
            self.last_iteration = self.modified;
        }
        pub fn mutate_large(&mut self, rng: &mut CustomRng) {
            let rn = rng.uniform_float();
            self.value = rn;
        }
        pub fn mutate_small(&mut self, current_iteration: i64, frng: f64, sigma: f64) {
            let idif = current_iteration - self.last_iteration;

            let samplestddistr = erf_inv(frng * 2.0 - 1.0) * 1.41421356237309504880;

            let effsigma = (idif as f64).sqrt() * sigma;
            let mut value = samplestddistr * effsigma;
            value += self.value;
            value -= value.floor();

            self.value = value;
            //  self.last_iteration = current_iteration;
        }
    }

    pub struct SamplerMetropolis {
        current_iter: i64,
        last_large_step_iteration: i64,
        large_step_prob: f64,
        islarge_step: bool,
        sigma_offset: f64,
        rng: CustomRng,
        pub samples: Vec<MetropolisSample>,
        pub streamid: i64, //if streamid is path light, path cam, connect
        pub streamcount: i64,
        pub sampleidx: i64,
    }
    impl SamplerMetropolis {
        pub fn new() -> Self {
            let mut custom_rng = CustomRng::new();
            custom_rng.set_sequence(0);
            SamplerMetropolis {
                current_iter: 0,
                last_large_step_iteration: 0,
                large_step_prob: 0.3,
                islarge_step: false,
                sigma_offset: 0.1,
                rng: custom_rng,
                samples: vec![],
                streamid: 0,
                streamcount: 3,
                sampleidx: 0,
            }
        }
        pub fn from_args(mutsperpixel_spp: u64, seed_idx: u64) -> Self {
            let mut custom_rng = CustomRng::new();
            custom_rng.set_sequence(seed_idx);
            SamplerMetropolis {
                current_iter: 0,
                last_large_step_iteration: 0,
                large_step_prob: 0.3,
                islarge_step: true,
                sigma_offset: 0.01,
                rng: custom_rng,
                samples: vec![],
                streamid: 0,
                streamcount: 3,
                sampleidx: 0,
            }
        }
        pub fn get1d(&mut self) -> f64 {
            let idx = self.get_idx();
            if idx >= self.samples.len() as i64 {
                self.samples
                    .resize((idx + 1) as usize, MetropolisSample::default())
            }
            self.new_sample(idx);
            self.samples[idx as usize].value
        }
        pub fn get2d(&mut self) -> (f64, f64) {
            (self.get1d(), self.get1d())
        }
        pub fn get_idx(&mut self) -> i64 {
            let idx = self.streamid + self.streamcount * self.sampleidx;
            self.sampleidx += 1;
            return idx;
        }
        pub fn start_iter(&mut self) {
            self.current_iter += 1;
            let rngnum = self.rng.uniform_float();
            // println!("  start_iter {}", rngnum);
            self.islarge_step = rngnum < self.large_step_prob;
            if self.islarge_step {
                // println!("      large step ON  {}", rngnum);
            }
        }
        pub fn new_sample(&mut self, i: i64) {
            // if i >= self.samples.len() as i64 { self.samples.push(MetropolisSample::default())}
            // let mut Xi = &mut  self.samples.last().unwrap();
            // Xi.last_iteration=10;
            let mut Xi = &mut self.samples[i as usize];

            if (Xi.last_iteration < self.last_large_step_iteration) {
                Xi.value = self.rng.uniform_float();
                Xi.last_iteration = self.last_large_step_iteration;
            }
            Xi.store();

            if self.islarge_step {
                Xi.mutate_large(&mut self.rng);
            } else {
                Xi.mutate_small(
                    self.current_iter,
                    self.rng.uniform_float(),
                    self.sigma_offset,
                );
            }
            Xi.last_iteration = self.current_iter;
            self.samples[i as usize] = Xi.clone();
        }

        pub fn accept(&mut self) {
            if self.islarge_step {
                self.last_large_step_iteration = self.current_iter;
            }
        }
        pub fn reject(&mut self) {
            self.samples.iter_mut().for_each(|f| {
                if self.current_iter == f.last_iteration {
                    f.rollback();
                }
            });
            self.current_iter -= 1;
        }
        pub fn set_channel(&mut self, ch: i64) {
            self.streamid = ch;
            self.sampleidx = 0;
        }
    }

    pub enum CHANNELS {
        CH_CAM = 0,
        CH_LIGHT = 1,
        CH_MERGE = 2,
    }
    const NCHS: u8 = 3;
    pub trait TraitMltSampler {
        fn accept(&mut self);
        fn reject(&mut self);
        fn lightstream(&mut self);
        fn mergestream(&mut self);
        fn camerastream(&mut self);
    }
    impl TraitMltSampler for SamplerMetropolis {
        fn accept(&mut self) {}
        fn reject(&mut self) {}
        fn camerastream(&mut self) {
            self.set_channel(CHANNELS::CH_CAM as i64)
        }
        fn lightstream(&mut self) {
            self.set_channel(CHANNELS::CH_LIGHT as i64)
        }
        fn mergestream(&mut self) {
            self.set_channel(CHANNELS::CH_MERGE as i64)
        }
    }

    impl Sampler for SamplerMetropolis {
        fn get1d(&mut self) -> f64 {
            self.get1d()
        }
        fn get2d(&mut self) -> (f64, f64) {
            self.get2d()
        }
        fn get_current_pixel(&self) -> (u32, u32) {
            (0, 0)
        }
        fn get_dims(&self) -> (Point2i, Point2i) {
            (Point2i::new(0, 0), Point2i::new(0, 0))
        }
        fn get_spp(&self) -> u32 {
            0
        }
        fn has_samples(&self) -> bool {
            false
        }
        fn start_next_sample(&mut self) -> bool {
            false
        }
        fn start_pixel(&mut self, p: Point2i) {}
        fn start_stream(&mut self, channel: CHANNELS) {
            panic!("")
            // //    let stridx =   channel as i64;
            // self.streamid = channel as i64;
            // self.sampleidx = 0;
        }
    }
    trait SamplerTest {
        fn get1d(&mut self) -> f64;
    }
    impl<SamplerMetropolis: Sampler> Sampler for &mut SamplerMetropolis {
        fn get1d(&mut self) -> f64 {
            self.get1d()
        }
        fn get2d(&mut self) -> (f64, f64) {
            self.get2d()
        }
        fn get_current_pixel(&self) -> (u32, u32) {
            (0, 0)
        }
        fn get_dims(&self) -> (Point2i, Point2i) {
            (Point2i::new(0, 0), Point2i::new(0, 0))
        }
        fn get_spp(&self) -> u32 {
            0
        }
        fn has_samples(&self) -> bool {
            false
        }
        fn start_next_sample(&mut self) -> bool {
            false
        }
        fn start_pixel(&mut self, p: Point2i) {}
        fn start_stream(&mut self, channel: CHANNELS) {
            panic!("")
            //    let stridx =   channel as i64;
            //    ( *self.streamid) = channel as i64;
            //     self.sampleidx = 0;
        }
    }
    

    /**
     * 
     * weights para calcular en paralelo la cdf bootstrap.
     * la granularidad es mala : vlocks mantine un solo lock. supone
     * que cuando se hace un Load/store  todo el array queda bloqueado.
     * campiar la estrategia por algo mas granular como un lock por segmento
     */
    pub struct Weights {
        pub vlocks: RwLock<Vec<f64>>,
    }

    #[derive(Debug, Clone)]
    pub struct WeightsSpan {
        ith_bootstap: i64,
        range: (i64, i64),
        v: Vec<f64>,
    }
    impl WeightsSpan {
        pub fn add(&mut self, ith: usize, f: f64) {
            self.v[ith] = f;
        }
    }
    impl Weights {
        pub fn new(sizerange: usize) -> Self {
            Weights {
                vlocks: RwLock::new(vec![0.0; sizerange]),
            }
        }
        /**
         *1. FASE SCATTER
         * es llamado en el incio del proceso.
         * asocia a cada segemento de tamaño 0 depth un rango
         * que es la posicion en el array global(la shared memory)
         * cada  WeightsSpan reprentara un bucket de la shared mem
         */
        pub fn init(&self, ith_bootstap: i64, range: &(i64, i64)) -> WeightsSpan {
            let range_span = (range.1 + 1 - range.0) as usize;
            let mut span_contens = vec![0.0; range_span];
            //    let tstv= span_contens.iter().enumerate().map(|o|{o.0 as f64}).collect();
            WeightsSpan {
                ith_bootstap,
                range: *range,
                v: span_contens,
            }
        }

        /**2. * FASE GATHER
         * la estrategia general es scatter / gather.
         * ese metodo produce la fase gather.
         * bloquea el array compartido y vuelca el contenido 
         * del ws: WeightsSpan con sus  coordenadas asociadas
         * en el vlocks
         * 
         */
        pub fn merge(&self, ws: WeightsSpan) {
            // println!("ws len {:?}", ws.v.len());

            for i in ws.range.0 as usize..=ws.range.1 as usize {
                let map_to_0 = i - ws.range.0 as usize;

                assert!(ws.v.len() == 6);
                assert!((ws.range.1 + 1 - ws.range.0) == 6);
                self.vlocks.write().unwrap()[i] = ws.v[map_to_0];
            }
            //   (&mut self.vlocks.write().unwrap()[])= &mut ws.v[..];
        }
    }

    pub struct MetropolisIntegrator<S: Sampler> {
        scene: Scene1,
        sampler: S,
        film: pararell::Film1,
        camera: PerspectiveCamera,
        config: Config,
        nbootstraps: i64,
        nchains : i64,
        maxdepths: i64,
        mutsperpixel_spp: i64,
        pub weights: Weights, // esta en pub para facilidad de debug
        ws: Vec<f64>,
    }

    impl<S: Sampler> MetropolisIntegrator<S> {
        pub fn new(
            scene: Scene1,
            sampler: S,
            film: pararell::Film1,
            camera: PerspectiveCamera,
            config: Config,
            nbootstraps: i64,
            maxdepths: i64,
            nchains :i64,
        ) -> Self {
            
            MetropolisIntegrator {
                scene,
                sampler,
                film,
                camera,
                config,
                nbootstraps, //10,// 100.000
                maxdepths: 5,
                mutsperpixel_spp: 100,
                weights: Weights::new((nbootstraps * (maxdepths + 1)) as usize),
           nchains:1000,
                ws: vec![], 
            }
        }
        pub fn from_standard_config(
            scene: Scene1,
            sampler: S,
             
            camera: PerspectiveCamera,
            config: Config,
            nbootstraps: i64,
            maxdepths: i64,
            nchains :i64,
            mutsperpixel_spp:i64,
        ) -> Self { 
            let mut film = pararell::Film1::new(BBds2i::from(
                BBds2f::<f64>::new(Point2::new(0.0, 0.0),Point2::new( scene.width as f64,  scene.height as f64),)),
                FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0, 2.0), 3.0)));

            MetropolisIntegrator {
                scene,
                sampler,
                film,
                camera,
                config,
                nbootstraps, //10,// 100.000
                maxdepths: 5,
                mutsperpixel_spp ,
                nchains ,
                weights: Weights::new((nbootstraps * (maxdepths + 1)) as usize),
                ws: vec![], 
            }
        }
        pub fn compute_nboots(&self, maxdepths: i64) -> i64 {
            (maxdepths + 1) * self.nbootstraps
        }
        /**
         * 
         * usa un rng del stram sampler.
         * proporciona la estrategia para bdpt
         */
        pub fn compute_strategies(
            depth: i64,
            sample1d: f64,
        ) -> (
            i64, // nlights s
            i64, //ncamera t
            i64, // nstrategies
        ) {
            // strategies

            if depth == 0 {
                (
                    0, // s lights
                    2, // camera t
                    1,
                ) // strategies
            } else {
                let strategies = depth + 2;
                let s_lights = ((sample1d * (strategies as f64)) as i64).min(strategies - 1);
                let t_camera = strategies - s_lights;
                (s_lights, t_camera, strategies)
            }
        }
        #[inline]
        pub fn get_config(&self) -> &Config {
            &self.config
        }
        #[inline]
        pub fn computeTotalMutationsInAreaFilm(&self) -> i64 {
            self.film.area() * self.mutsperpixel_spp
        }
        #[inline]
        pub fn computeMutationsPerChain(
            &self,
            ichain: i64,
            nChains: i64,
            nTotalMutationsInArea: i64,
        ) -> i64 {
            ((ichain + 1) * (nTotalMutationsInArea / nChains)).min(nTotalMutationsInArea)- ichain * (nTotalMutationsInArea / nChains)
        }
        /**
         *
         * scene: &Scene1,
        pfilm: &(f64, f64),
        paths: &mut Vec<PathTypes>,
        camera: PerspectiveCamera,
        sampler: &mut S,
        search_depth: usize,
         */
        pub fn mlt_path<Sm: Sampler + TraitMltSampler>(
            &self,

            i_mutationchain: i64,
            samplermlt: &mut Sm,
            pfilm: &mut (f64, f64),
            search_depth: i64,
        ) -> Srgb {
           
            let scene = self.get_scene();
            let cameraInstance = self.get_camera();
            let film = self.get_film();
            samplermlt.camerastream();

            let mut stnstrategies; // =( 0_i64,  0_i64,  0_i64);
            if (search_depth != 0) {
                let psample = samplermlt.get1d();

                stnstrategies = Self::compute_strategies(search_depth, psample);

                //  println!("nstrategies: {} s {}, t: {}, psample {} ",stnstrategies.2,  stnstrategies.0, stnstrategies.1 , psample);
            } else {
                stnstrategies = Self::compute_strategies(search_depth, 0.0);
                //    println!("nstrategies: {} s {}, t: {},  ",stnstrategies.2,  stnstrategies.0, stnstrategies.1 );
            }
            let psample2d = samplermlt.get2d();
            //    let prasterfilm : (f64, f64) = film.bounds.lerp(&psample2d);
            let prasterfilm: (f64, f64) = film.bounds.lerp(&psample2d);

            *pfilm = prasterfilm;

            let mut pathcamera: Vec<PathTypes> = vec![];
            let depth_t = mlt_camera_path(
                &scene,
                &prasterfilm,
                &mut pathcamera,
                *cameraInstance,
                samplermlt,
                stnstrategies.1 as usize,
            );
            if (pathcamera.len() as i64 != stnstrategies.1) {
                // println!(" mlt_camera_path :: NOVALID s, t");
                return Srgb::new(0.,0.0,0.);
            }
            samplermlt.lightstream();

            //   <SamplerMetropolis as TraitMltSampler>::lightstream(samplermlt);
            let mut pathlight: Vec<PathTypes> = vec![];
            let depth_s_light =
                mlt_light_path(&scene, &mut pathlight, samplermlt, stnstrategies.0 as usize);
                if ( pathlight.len() as i64 != stnstrategies.0) {
                    // println!(" mlt_light_path :: NOVALID s, t");
                    return Srgb::new(0.,0.0,0.);
                }

            samplermlt.mergestream();

            let Lpath = mlt_merge_path(
                scene,
                stnstrategies.0 as i32,
                stnstrategies.1 as i32,
                &mut pathcamera,
                &mut pathlight,
                samplermlt,
                cameraInstance,
                &prasterfilm,
            );
            let Lpath = Srgb::new(
                Lpath.red * (stnstrategies.2 as f32),
                Lpath.green * (stnstrategies.2 as f32),
                Lpath.blue * (stnstrategies.2 as f32),
            );
            Lpath
        }

        fn inner_compute_weights(
            ith_depth: i64,
            rng_cdf_bucket_entry: i64,
            scene: &Scene1,

            camera: PerspectiveCamera,

            film: &pararell::Film1,
            cameraInstance: &PerspectiveCamera,
        ) -> Option<f64> {
            let mut samplermlt = SamplerMetropolis::from_args(100, rng_cdf_bucket_entry as u64);
            <SamplerMetropolis as TraitMltSampler>::camerastream(&mut samplermlt);
            let mut stnstrategies; // =( 0_i64,  0_i64,  0_i64);
            if (ith_depth != 0) {
                let psample = samplermlt.get1d();

                stnstrategies = Self::compute_strategies(ith_depth, psample);

                //  println!("nstrategies: {} s {}, t: {}, psample {} ",stnstrategies.2,  stnstrategies.0, stnstrategies.1 , psample);
            } else {
                stnstrategies = Self::compute_strategies(ith_depth, 0.0);
                //    println!("nstrategies: {} s {}, t: {},  ",stnstrategies.2,  stnstrategies.0, stnstrategies.1 );
            }
            let psample2d = samplermlt.get2d();
            let prasterfilm: (f64, f64) = film.bounds.lerp(&psample2d);

            let mut pathcamera: Vec<PathTypes> = vec![];
            let depth_t = mlt_camera_path(
                &scene,
                &prasterfilm,
                &mut pathcamera,
                *cameraInstance,
                &mut samplermlt,
                stnstrategies.1 as usize,
            );
            if (pathcamera.len() as i64 != stnstrategies.1) {
                // println!(" NOVALID s, t");
                return None;
            }
            <SamplerMetropolis as TraitMltSampler>::lightstream(&mut samplermlt);
            let mut pathlight: Vec<PathTypes> = vec![];
            let depth_s_light = mlt_light_path(
                &scene,
                &mut pathlight,
                &mut samplermlt,
                stnstrategies.0 as usize,
            );
            if (pathlight.len() as i64 != stnstrategies.0) {
                return None;
            }
            

            <SamplerMetropolis as TraitMltSampler>::mergestream(&mut samplermlt);
            let Lpath = mlt_merge_path(
                scene,
                stnstrategies.0 as i32,
                stnstrategies.1 as i32,
                &mut pathcamera,
                &mut pathlight,
                &mut samplermlt,
                cameraInstance,
                &prasterfilm,
            );
            let Lpatha = Srgb::new(
                Lpath.red * (stnstrategies.2 as f32),
                Lpath.green * (stnstrategies.2 as f32),
                Lpath.blue * (stnstrategies.2 as f32),
            );
        
            let Y = Lpatha.y();
            Some(Y)
        }

        /**
        *
        *  film.add_sample_mlt(
               if (accept > 0)
                   film.AddSplat(pProposed,
                                 LProposed * accept / LProposed.y());
               film.AddSplat(pCurrent, LCurrent * (1 - accept) / LCurrent.y());
           ) 
        */
       
        pub fn add_sample_mlt(
            &self,
            pProp: &(f64, f64),
            Lprop: Srgb,
            pCurr: &(f64, f64),
            Lcurr: Srgb,
            tile: &mut pararell::FilmTile,
            accept: f64,
        ) {
         
        //    println!("AddSplat ");
            if(accept>0.0){
                let lumProp = Lprop.y();
                let  ratioProp = (accept  / lumProp) as f32;
                let c = Srgb::new(Lprop.red * ratioProp , Lprop.green *ratioProp , Lprop. blue * ratioProp);
                  
              //   println!("     pProposed  {:?} LProposed {:?} ", pProp, c);
             tile. add_sample_mlt(pProp,  c);
           
              
            }
            let lumCurr = Lcurr.y();
              let  ratioCurr = ((1.0 - accept)  / lumCurr) as f32;
             let ccurrent =  Srgb::new(Lcurr.red * ratioCurr , Lcurr.green *ratioCurr , Lcurr.blue * ratioCurr);
           //  println!("     pCurrent  {:?} LCurrent {:?} ", pCurr, ccurrent);
           tile. add_sample_mlt(pCurr,  ccurrent);
            //  tile. add_sample_mlt(pProp,  c);
           //   println!("add_sampler  {:?} {:?} ", pCurr, Srgb::new(Lcurr.red * ratioCurr , Lcurr.red *ratioCurr , Lcurr.red * ratioCurr))
           
            
        }

         
        pub fn computeRanges(nbootstraps: i64, maxdepths: i64) -> Vec<(i64, i64, i64)> {
            (0..nbootstraps)
                .collect::<Vec<i64>>()
                .into_iter()
                .map(move |ith_bootstrap| {
                    let start = (maxdepths + 1) * ith_bootstrap;
                    let end = start + (maxdepths + 1);
                    // println!("{:?}",  (start, end-1));
                    (ith_bootstrap, start, end - 1)
                })
                .collect::<Vec<(i64, i64, i64)>>()
        }

        pub fn preprocess_par(&self) -> (Distribution1d, f64) {
            let timer = Instant::now();
            let nbootssamplessize = self.compute_nboots(self.maxdepths) as usize;

            let maxdepths = self.maxdepths;
            let nbootstraps = self.nbootstraps;

            let implementation = self;
            let film = implementation.get_film();
            let scene = implementation.get_scene();
            let mut weights = &implementation.weights;
            let camera = implementation.get_camera();

            crossbeam::scope(|scope| {
                let (sender, rvc) = crossbeam_channel::unbounded();
                let mut ws = weights;

                let film_ref = film;
                let scene_ref = scene;
                let cameraInstance = camera;

                scope.spawn(move |_| {
                    let sendwork = sender.clone();

                    for (ith_bootstap, start, end) in
                        Self::computeRanges(nbootstraps, maxdepths).into_iter()
                    {
                        let tile = ws.init(ith_bootstap, &(start, end));

                        sendwork
                            .send(tile)
                            .unwrap_or_else(|op| panic!("---sender error!!--->{}", op));
                    }

                    drop(sendwork);
                });

                for _ in 0..nbootstraps {
                    let rxclone = rvc.clone();

                    scope.spawn(move |_| {
                        let mut w = rxclone.recv().unwrap();
                        for ith_bucket in w.range.0..w.range.1 + 1 {
                            // tenemos que transfor  ith_bucket a depth (ith_depth - w.range.0)
                            let ith_depth = (ith_bucket - w.range.0);
                            let ith = w.ith_bootstap;
                            let rng_cdf_bucket_entry = ith * (maxdepths + 1) + ith_depth;

                            let yOP = Self::inner_compute_weights(
                                ith_depth,
                                rng_cdf_bucket_entry,
                                scene,
                                *cameraInstance,
                                film,
                                cameraInstance,
                            );
                            let Y = yOP.unwrap_or(0.);
                            //   println!("parar ithboot {}, ith_depth : {} , Y  {}",ith, ith_depth,Y );
                            w.add(ith_depth as usize, Y as f64);
                        }
                        ws.merge(w);
                    });
                }
            })
            .unwrap();
            println!(
                "pararell   cdf compute boots time : {} seconds",
                timer.elapsed().as_secs_f64()
            );
            let newws = self.weights.vlocks.read().unwrap().clone();

            let distribution = Distribution1d::new(&newws);
            let b = distribution.norm * ((self.maxdepths + 1) as f64);
            println!("B->{}", b);
            (distribution, b)
        }

        pub fn preprocess_ser(&mut self) -> (Distribution1d, f64) {
            let timer = Instant::now();
            let nbootssamplessize = self.compute_nboots(self.maxdepths) as usize;

            let scene = self.get_scene();
            let res = (scene.width as i64, scene.height as i64);
            let cameraInstance = self.get_camera();
            let film = self.get_film();
            

            let mut weights = vec![0.0; nbootssamplessize];

            let ws = Weights::new((self.nbootstraps * (self.maxdepths + 1)) as usize);

            let mut vs: Vec<WeightsSpan> = vec![];
            for (ith_bootstap, start, end) in
                Self::computeRanges(self.nbootstraps, self.maxdepths).into_iter()
            {
                let tile = ws.init(ith_bootstap, &(start, end));
                vs.push(tile);
            }

            for ith in 0..self.nbootstraps {
                for ith_depth in 0..=self.maxdepths {
                    let rng_cdf_bucket_entry = ith * (self.maxdepths + 1) + ith_depth;
                    if rng_cdf_bucket_entry==7 {
                        let hh = 1;
                    }
                    let opY = Self::inner_compute_weights(
                        ith_depth,
                        rng_cdf_bucket_entry,
                        scene,
                        *cameraInstance,
                        film,
                        cameraInstance,
                    );
                    let Y = opY.unwrap_or(0.);
                    // println!("bucket {} serial : ithboot {}, ith_depth : {} , Y  {}",rng_cdf_bucket_entry,ith, ith_depth,Y );
                    weights[rng_cdf_bucket_entry as usize] = Y as f64;
                    // vs[ith as usize].add(ith_depth as usize, Y);
                }
            }

            // falla el calculo de la cdf cuando se varia la resolucion:
            //     1. bajo la resolucion a 8x8
            //     2. bajo el numero de bootstrap  a 1000 o asi... y uso el codigo hecho por mi...
            
            // for span in vs.into_iter(){
            //     ws.merge(span);
            // }
            // for i in weights. iter().enumerate(){
            //   let y = self.weights.vlocks.read().unwrap()[i.0 ];
            //     println!("{} -> serial {}, pararel {}",i.0, *i.1,y);
            //     assert!(*i.1==y);

            // }

            let distribution = Distribution1d::new(&weights);
            let b = distribution.norm * ((self.maxdepths + 1) as f64);
            println!(" num bootstraps : {} ",self.nbootstraps ); 
            println!("      size of cdf    {} ",nbootssamplessize );
            println!("      cdf compute boots time: {} seconds ", timer.elapsed().as_secs_f64());
            println!("      B->{}", b);
            (distribution, b)
            
        }
        pub fn integrate(
            &mut self,
            // pfilm: &(f64, f64),
            // lights: &Vec<Light>,
            // scene: &Scene1,
            // depth: i32,
            // mut samplerhalton: &mut SamplerType,
        ) -> Srgb {
            let start = Instant::now();
            let (distribution,     normalization_constant) = self.preprocess_ser();
            let scene = self.get_scene();
            let res = (scene.width as i64, scene.height as i64);
            let film = self.get_film();
            let mut wholetile = film.init_tile(BBds2i::from(BBds2f::<f64>::new(
                Point2::new(0.0, 0.0),
                Point2::new(res.0 as f64, res.1 as f64),
            )));

            // el obj de esta semana es hacer esto serialmente.
            //     1. comprobar el sample de la distribution

            let nChains =  self.nchains;
            let nTotalMutationsInArea = self.computeTotalMutationsInAreaFilm();
            let mut checksum: i32 = 0;
            let mut pCheckSum: (f64, f64) = (0.0, 0.0);
            let mut LCheckSum: Srgb = Srgb::new(0., 0., 0.);
            for ithchain in 0..nChains {
                let nMutationsPerChain =self.computeMutationsPerChain(ithchain, nChains, nTotalMutationsInArea);
                // println!("ithchain {} nMutationsPerChain : {}",ithchain, nMutationsPerChain);
                let mut rndm = CustomRng::new();
                rndm.set_sequence(ithchain as u64);
                let nrng = rndm.uniform_float();
              
                let idx_boot = distribution.find(nrng);
                let depth = (idx_boot as i64) % (self.maxdepths + 1);
                let mut samplermlt =
                    SamplerMetropolis::from_args(self.mutsperpixel_spp as u64, idx_boot as u64);
                let mut pFilmCurrent: (f64, f64) = (0., 0.);

                // println!("{}",ithchain);
                checksum += idx_boot;
                let mut Lpathcurrent: Srgb =self.mlt_path(ithchain, &mut samplermlt, &mut pFilmCurrent, depth);
                //  println!("ithchain {} pfilm : {:?} LpathCurrent {:?}",ithchain, pFilmCurrent, Lpathcurrent);
                pCheckSum.0 += pFilmCurrent.0;
                pCheckSum.1 += pFilmCurrent.1;
                LCheckSum.blue += Lpathcurrent.blue;
                LCheckSum.red += Lpathcurrent.red;
                LCheckSum.green += Lpathcurrent.green;

                let mut acceptratioChcecksum: f64 = 0.;
                let mut pProposedCheckSum: (f64, f64) = (0.0, 0.0);
                let mut LProposedCheckSum: Srgb = Srgb::new(0., 0., 0.);
 
                //

                //
                //         // for i_mutationchain in  0..nMutationsPerChain{
                for i_mutationchain in 0.. nMutationsPerChain{
                    // println!("i_mutation {}",i_mutationchain);
                    if i_mutationchain == 2{
                        let kkk = 1;
                    }
                    samplermlt.start_iter(); 

                    let mut pFilmother: (f64, f64) = (0., 0.);
                    let mut Lpathother: Srgb =
                        self.mlt_path(i_mutationchain, &mut samplermlt, &mut pFilmother, depth);
                    let probacceptation = (Lpathother.y() / Lpathcurrent.y()).min(1.0);

                    acceptratioChcecksum += probacceptation;
                    pProposedCheckSum.0 += pFilmother.0;
                    pProposedCheckSum.1 += pFilmother.1;
                    LProposedCheckSum.blue += Lpathother.blue;
                    LProposedCheckSum.red += Lpathother.red;
                    LProposedCheckSum.green += Lpathother.green;
                    let lumCurr = Lpathcurrent.y();
                    let ratioCurr = ((1.0 - probacceptation) / lumCurr) as f32;
                    let Lc = Srgb::new(
                        Lpathcurrent.red * ratioCurr,
                        Lpathcurrent.green * ratioCurr,
                        Lpathcurrent.blue * ratioCurr,
                    );
 

                    if probacceptation > 0.0 {
                        let lumProp = Lpathother.y();
                        let ratioProp = (probacceptation / lumProp) as f32;
                        let Lp = Srgb::new(
                            Lpathother.red * ratioProp,
                            Lpathother.green * ratioProp,
                            Lpathother.blue * ratioProp,
                        );

                    
                    }

                    //     //     tile.add_sample(pProp,  Srgb::new(Lprop.red * ratioProp , Lprop.red *ratioProp , Lprop.red * ratioProp));
                     self.add_sample_mlt(&pFilmother, Lpathother, &pFilmCurrent, Lpathcurrent, &mut wholetile, probacceptation);

                    let acceptProb = rndm.uniform_float();
                    if acceptProb < probacceptation {
                        std::mem::swap(&mut Lpathcurrent, &mut Lpathother);
                        std::mem::swap(&mut pFilmCurrent, &mut pFilmother);
                        // Lpathcurrent=Lpathother;
                        // pFilmCurrent=pFilmother;
                        // damos por bueno el trabajo hecho. aceptamos los nuevos samples.
                        // aqui esta el problema de estp... no hay manera de hacerlo paralelo.
                        // los cambios deben ser realizados en en funcion de lo anterior
                        samplermlt.accept();
                    } else {
                        // rechaza el ultimo trabajo hecho, recuperando los samples que habia
                        // antes de empezar la iteracion
                        samplermlt.reject();
                    }
                    
                  //     self.acceptOrReject(&pFilmCurrent,Lpathcurrent,&pFilmother, Lpathother,samplermlt, rndm,probacceptation );
                }
                // println!("ithchain  {}",ithchain );
                // println!("  acceptratiochecksum   {}  ", acceptratioChcecksum);
                // println!("  pProposedCheckSum     {:?}", pProposedCheckSum);
                // println!("  LProposedCheckSum     {:?}", LProposedCheckSum);
           
                pCheckSum.0 += pProposedCheckSum.0;
                pCheckSum.1 += pProposedCheckSum.1;
                
                LCheckSum.red += LProposedCheckSum.red;
                LCheckSum.green += LProposedCheckSum.green;
                LCheckSum.blue += LProposedCheckSum.blue;
            }
            //   println!("  pChecksum : {:?} LCheckSum {:?}", pCheckSum, LCheckSum);
              wholetile.mult_by_constant(  normalization_constant  as f32 / self.mutsperpixel_spp as f32);
             film.merge_splat_tile(&wholetile);
 
            let conf = self.get_config();
        
            let filename : Option<String>  =  conf.filename.clone();
            if filename.is_some(){
                let directory =  conf.directory.as_ref().unwrap();
                film.commit_and_write(&directory,&filename.unwrap(), false).unwrap();
                  let configtoprint = Config::from_config(self. get_config(),  start.clone());
                //   configtoprint.save();
                println!("{}",    serde_json::to_string(&configtoprint).unwrap() );
            }
        
           
            Srgb::default()
        }
        pub fn get_res(&self) {}
        pub fn get_scene(&self) -> &Scene1 {
            &self.scene
        }
        pub fn get_spp(&self) -> u32 {
            self.sampler.get_spp()
        }
        pub fn get_film(&self) -> &pararell::Film1 {
            &self.film
        }
        pub fn get_camera(&self) -> &PerspectiveCamera {
            &self.camera
        }
        pub fn get_sampler(&self) -> &S {
            &self.sampler
        }
    }
}

#[test]
pub fn meto_debug() {
    metro_main();
}
pub fn metro_main() {
    let res = (512,512);
     let config = Config::from_args(
        10,
        res.0 as i64,
        res.1 as i64,
        1,
        Some("newdir".to_string()),
        Some("mlt_disk.png"),
    );


     
    let scene = Scene1::make_scene(
        res.0,
        res.1,
        vec![Light::PointLight(PointLight {
            iu: 4.0,
            positionws: Point3::new(0.0000, 1.0, 0.0000),
            color: Srgb::new(5.0,5.0,5.0),
        })],
        vec![
            PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.50, 0., 2.0),0.5,MaterialDescType::PlasticType(Plastic::from_albedo(1., 1., 0.0)),)),
             PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.50, 0., 2.0),0.5,MaterialDescType::PlasticType(Plastic::from_albedo(0., 1., 1.0)),))
        ],
        1,
        1,
    );
    // mira el point light : es absurdo!
    // cuadno aumtento la luz el B->aumenta. de modo uso la ratio b/num_mutattions
    // puedo usar solo una bolita en el centro 
    // quizas tengo que usar el mlt de pbrt a ver que tal va

    let samplermlt = SamplerMetropolis::new();
    let cameraInstance = PerspectiveCamera::from_lookat(
        Point3f::new(0.0, 0.000, 0.0),
        Point3f::new(0.0, 0.0, 1.00),
        1e-2,
        100.0,
        45.0,
        (res.0 as u32, res.1 as u32),
    );
   
    let mut mlt = MetropolisIntegrator::from_standard_config(
        scene,
        samplermlt,
        cameraInstance,
        config,
        100000, //100000,
        5,
        1000,1000,
    );
    mlt.integrate();

    //    tengo que comprobar si el global memory se llena ordenadamente
    //    tengo que mirar cuando no se aujusta el attay
    //    empezar a mirar lo de meter el sample y las functiones
    //     mlt.preprocess_par();
    //    mlt.preprocess_ser();

    return;
    // mlt.weights.vlocks.read().unwrap().iter().enumerate().for_each(|s|{println!("->{:?}", s);});

    // let mut sampler = SamplerMetropolis::new();

    // sampler.start_iter();
    // sampler.start_stream(CHANNELS::CH_CAM);
    // for _ in 0..32 {
    //     let o0 = sampler.get1d();
    //     // println!("{}", o0);
    // }
    // sampler.start_stream(CHANNELS::CH_LIGHT);
    // for _ in 0..32 {
    //     let o0 = sampler.get1d();
    //     // println!("{}", o0);
    // }

    // sampler.start_stream(CHANNELS::CH_MERGE);
    // for _ in 0..32 {
    //     let o0 = sampler.get1d();
    //     // println!("{}", o0);
    // }
    // for p in sampler.samples.iter() {
    //     println!("{}", p.value);
    // }

    // sampler.reject();
    // for p in sampler.samples.iter() {
    //     println!("{}", p.value);
    // }
    // {
    //     let scene = Scene1::make_scene(1, 1, vec![], vec![], 1, 1);

    //     let mut pathlight: Vec<PathTypes> = vec![];
    //     let mut pathcamera: Vec<PathTypes> = vec![];
    //     let mut samplermlt = SamplerMetropolis::new();
    //     <SamplerMetropolis as TraitMltSampler>::lightstream(&mut samplermlt);
    //     samplermlt.get1d();
    //     <SamplerMetropolis as TraitMltSampler>::camerastream(&mut samplermlt);
    //     samplermlt.get1d();
    //     <SamplerMetropolis as TraitMltSampler>::mergestream(&mut samplermlt);
    //     samplermlt.get1d();
    //     <SamplerMetropolis as TraitMltSampler>::accept(&mut samplermlt);
    //     samplermlt.get1d();

    //     samplermlt.reject();

    //     // no he hecho el build... falla seguro el vtx... sale en rojo
    //     // mira el folip:k
    //     // let [s, t] = compute_strategies(depth, sample1d)
    //     let (s_lights, t_camera, nstrategies) = MetropolisIntegrator::compute_strategies(1, 1.0);

    //     let pfilm = (0.0, 0.0);
    //     mlt_light_path(&scene, &mut pathlight, &mut samplermlt, 1);
    //     mlt_camera_path(
    //         &scene,
    //         &pfilm,
    //         &mut pathcamera,
    //         cameraInstance,
    //         &mut samplermlt,
    //         1,
    //     );
    // }

    // let sampleru  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (0 as u32, 01 as u32)),1 as u32,false,Some(0)) );
}

#[test]
pub fn sampler_mlt_test() {
    let mut sampler = SamplerMetropolis::from_args(100, 0);

    for iter_str in 0..100 {
        println!("iter_str : {:?} ", iter_str);
        sampler.start_iter();
        sampler.camerastream();
        for _ in 0..1 {
            let o0 = sampler.get2d();
            // println!("{}", o0);
        }
        sampler.lightstream();
        for _ in 0..1 {
            let o0 = sampler.get2d();
            // println!("{}", o0);
        }

        sampler.mergestream();
        for _ in 0..1 {
            let o0 = sampler.get2d();
            // println!("{}", o0);
        }
    }
    // for p in sampler.samples.iter() {
    //     println!("{}", p.value);
    // }
}
