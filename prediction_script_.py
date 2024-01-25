from classifier import image_classifier
import time


sample_image_pixels = "0.6021333020115345 0.6074023027599443 0.6509381044713545 0.5790864001827137 0.607933950664585 0.6888653021601379 0.5549427955466875 0.6060288105440836 0.7254835888965914 0.5338598293611303 0.6015485883644024 0.7533248178486014 0.5087018804329441 0.5949589226589461 0.7810283793025073 0.4999938155427259 0.5928463468808937 0.7901604646496623 0.48954424747903036 0.5903225155370523 0.8008965444196507 0.46899148477118985 0.5856531139364618 0.8223609726393927 0.46470643591191796 0.584401876314397 0.8275615408843319 0.47040788701592406 0.5867467562275779 0.8235224252796521 0.5899020946176752 0.6057363198949289 0.6418599734206509 0.5914935723455895 0.6128921918418123 0.6567930887894087 0.589886199290011 0.6205541176268133 0.6779726655355601 0.5898956573196608 0.6283401494281992 0.6961461504383784 0.5763540118691943 0.6339729988403968 0.7297858548148124 0.5742394929336165 0.6267588992400388 0.7204417119760285 0.5840927003179957 0.6317897323350862 0.7204200089960276 0.5728512765961266 0.6495559203489102 0.7788405909489456 0.5445403075869634 0.6401339722227019 0.7942246246364223 0.5323399869379787 0.6394875390280808 0.8069762371672727 0.5952471180622371 0.585526619501979 0.5673201954303784 0.6077755433343647 0.5953763971890857 0.5745731339605454 0.6177431915781236 0.6102186723272078 0.5934876156720226 0.6206477934347714 0.6311361466147177 0.6322622897672158 0.615231076225224 0.6467509532490401 0.6746397037099942 0.6190451443563733 0.6311130938958511 0.6520961803121855 0.6813430912995023 0.6833439089893337 0.6970081253661509 0.6986233502775295 0.7103381986199799 0.7334187310623772 0.6431847325794859 0.6765606611417331 0.7094121825907413 0.623640221594862 0.6818155583126854 0.7357840411109011 0.600037572957878 0.5458255083868253 0.4736888371520549 0.6137472591333355 0.5822005316278656 0.5394958277212429 0.6092377768929016 0.5947821706125679 0.5754819821436223 0.6975435939710124 0.6898428190914971 0.6769223620701723 0.7236027332515036 0.7249460378609776 0.728669350829589 0.7620903513125027 0.7656940155513678 0.7877451610103545 0.7403028426870051 0.7382778505846144 0.7443615380705404 0.6699699598497972 0.668411698752996 0.6647538413251529 0.6690729122093109 0.6823802012021435 0.6638884647792748 0.6622940973814793 0.6930334513040022 0.6830717067328669 0.5845441338608139 0.5225657909432422 0.4417391732430907 0.5918306424199057 0.5564942075311892 0.5131943813509312 0.6066062841095061 0.6035390913088423 0.6154518675519528 0.7085677853902131 0.7118403211566617 0.7353976050063853 0.7726360872203856 0.7788389320175974 0.8079367825503584 0.769128820101954 0.7729785257939935 0.7974132966885967 0.7423975325751532 0.7339330366881226 0.7234431536332044 0.6321949272431836 0.6254630311908963 0.5937763323503912 0.6552738475182562 0.6734653782452846 0.6352534125036687 0.6626879450701002 0.6977082644891394 0.6652385547154988 0.5709467464850061 0.49306951620993966 0.3993441604942771 0.5680860177997302 0.5140746916515624 0.45447775129215456 0.5153972140648319 0.5040316778052382 0.505446044366334 0.4384439187758406 0.42766388528030647 0.42825657704668174 0.5912578750406693 0.5758881944016087 0.573378980247176 0.6588954729762259 0.6480947938293682 0.6473030587638628 0.6669370307224801 0.6435942524656291 0.6148806169297791 0.6139181370654695 0.5770606382459057 0.5081097265692569 0.6274814015040986 0.6098979248469325 0.5429356502947771 0.6439142209959279 0.6596243618484744 0.6080863458125798 0.6084792724007801 0.5295098544976086 0.44228419078737086 0.5832980766549667 0.5321028647191186 0.4796010897412397 0.4787398020370204 0.44379690004149686 0.41307031238917286 0.4207893214620765 0.37713420658201374 0.33153538925893783 0.5292659314778818 0.4670617371365854 0.395920238910629 0.5919833628491525 0.5418336522461967 0.4861283880223553 0.5847540373366275 0.5276327676847644 0.4601105557398073 0.6039644437833988 0.5484249139971145 0.4752712125478575 0.589045602079449 0.5281400382609933 0.4395299984198826 0.5987985015696133 0.5533355757281838 0.47250318694196003 0.5979634389230225 0.525440480196683 0.44692008673644057 0.5908538539071357 0.5240785677281142 0.45181744285665254 0.5802714983761029 0.5183065973214376 0.4517199162176374 0.5666107805037872 0.49035734635289757 0.40314919519610026 0.5776310611492811 0.5043416622251854 0.417683323851577 0.6101441284581426 0.5481771491775145 0.4755533783747474 0.6042818347861427 0.5308205837417103 0.4407062354761716 0.6482928308070629 0.5863132670926947 0.5083003643590968 0.6219791861516702 0.5517985959561561 0.4610436402227423 0.5872232601554739 0.5162731246743223 0.4238757320545602 0.6101831931447611 0.5376771403824203 0.4515913396707983 0.6026983792916853 0.5280484264122484 0.437274915154247 0.5949354531553426 0.5207260883715841 0.43197643151747306 0.5970681285282762 0.5217704131580361 0.4301578699811258 0.6121278266150803 0.5388988399300149 0.4487255646014703 0.6394926268032257 0.5721981917292243 0.48867538302287006 0.6360791146710028 0.5615357606970823 0.46761255170417276 0.6486052862569821 0.5754445135638064 0.48274500663937403 0.6501063845583861 0.5760160599881864 0.482274834947902 0.6271455013477518 0.5518672612545392 0.4574734390594708 0.6602891819883829 0.5773773500968087 0.46978173943595725 0.6566880073756776 0.5723185124588863 0.46247191808617727 0.6627632552694255 0.5785146809538706 0.4702920436763496 0.6693310035543363 0.5860028485564267 0.47919038377585954 0.6738757442671648 0.5897531940173737 0.48522459923103634 0.679288309513361 0.5962241063897707 0.4926181076152839 0.676450722192196 0.5924156076293269 0.4884911493562605 0.6715853147107026 0.588996265392031 0.48670125734978753 0.6715632806312548 0.5901060033553677 0.4881173506941211 0.6664781916327382 0.5848574088944194 0.48072541960626425"

start = time.time()
for _ in range(5000):
    processed_image_pixels = list(map(float, sample_image_pixels.split(' ')))
    image_expended_dim, result, predicted_class, predicted_class_name = image_classifier(processed_image_pixels)
stop = time.time()
print('Process Time Taken: ', stop-start)





