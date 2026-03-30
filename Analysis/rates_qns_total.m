function dxdt = rates_qns_total(t, x, params)
tmp = qns_perturbation_breakdown(t, x, params);
dxdt = tmp.rates.total;
end